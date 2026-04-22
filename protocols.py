import json
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Tuple, Type, TypedDict, Union

from openai import OpenAI

if TYPE_CHECKING:
    from google.genai import client as _genai_client
    import anthropic as _anthropic

# Connection type aliases used by BulletStore.http_client.
HostPort = Tuple[str, str]          # (host, port) — e.g. ChromaDB
ConnectionString = str              # Full DSN — e.g. postgresql://user:pw@host/db


class QueryResult(TypedDict):
    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict]]


# ── Abstract base classes ─────────────────────────────────────────────────────

class AIClient(ABC):
    """Abstract AI provider — concrete implementations: OpenAI, Google, Anthropic."""

    def __init__(self) -> None:
        model_name = os.getenv("RESUME_AI_MODEL_NAME")
        if not model_name:
            raise Exception("RESUME_AI_MODEL_NAME must be defined in the script's .env file")
        self._model = model_name

    def __enter__(self) -> "AIClient":
        return self

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the inner client."""
        ...

    @staticmethod
    def get_system_prompt_jd_extraction() -> str:
        from prompts import JD_EXTRACTION_PROMPT
        return JD_EXTRACTION_PROMPT

    @staticmethod
    def get_system_prompt_resume_generation() -> str:
        from prompts import RESUME_GENERATION_PROMPT
        return RESUME_GENERATION_PROMPT

    @abstractmethod
    def parse_jd(self, text: str) -> dict:
        """Extract structured requirements from a job description.

        Implementations should use get_system_prompt_jd_extraction() as the
        system role content and text as the user role content.
        """
        ...

    @abstractmethod
    def complete_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> dict:
        """Run a chat completion and return the parsed JSON response.

        Falls back to self._model when model is not supplied. Implementations
        should inject get_system_prompt_resume_generation() as the system role
        when no system message is present in messages.
        """
        ...


class BulletStore(ABC):
    """Abstract vector store — concrete implementations: ChromaDB, pgvector."""

    def __init__(self) -> None:
        required = {
            "ai_platform": "RESUME_AI_PLATFORM",
            "model_name": "RESUME_AI_MODEL_NAME",
            "collection_name": "RESUME_COLLECTION_NAME",
        }
        values: dict[str, str] = {}
        for var, env_key in required.items():
            val = os.getenv(env_key)
            if not val:
                raise Exception(f"{env_key} must be defined in the script's .env file")
            values[var] = val

        db_host = os.getenv("RESUME_DB_HOST")
        if db_host:
            db_port = os.getenv("RESUME_DB_PORT")
            connect_with: Union[HostPort, ConnectionString] = (
                (db_host, db_port) if db_port else db_host
            )
        else:
            connection_string = os.getenv("RESUME_DB_CONNECTION_STRING")
            if not connection_string:
                raise Exception(
                    "Either RESUME_DB_HOST or RESUME_DB_CONNECTION_STRING "
                    "must be defined in the script's .env file"
                )
            connect_with = connection_string

        self.embedding_fn = self.embedding_function(values["ai_platform"], values["model_name"])
        self._client = self.http_client(connect_with)
        self._collection = self.get_or_create_collection(self._client, values["collection_name"])

    def __enter__(self) -> "BulletStore":
        return self

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the inner database client."""
        ...

    @staticmethod
    def embedding_function(ai_platform: str, model_name: str) -> Any:
        """Return the ChromaDB embedding function for the given AI platform."""
        from chromadb.utils import embedding_functions
        if ai_platform == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(model_name=model_name)
        if ai_platform == "google":
            return embedding_functions.GoogleGeminiEmbeddingFunction(model_name=model_name)
        if ai_platform == "anthropic":
            return embedding_functions.VoyageAIEmbeddingFunction(model_name=model_name)
        raise Exception(
            f"Unknown ai_platform '{ai_platform}'. "
            "Valid choices: openai, google, anthropic."
        )

    @staticmethod
    @abstractmethod
    def http_client(connect_with: Union[HostPort, ConnectionString]) -> Any:
        """Return a client connected to the vector database."""
        ...

    @abstractmethod
    def get_or_create_collection(self, client: Any, name: str) -> Any:
        """Return the named collection, creating it if it does not exist.

        Implementations should assign self.embedding_fn to the collection.
        """
        ...

    @abstractmethod
    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Index documents with their ids and metadata."""
        ...

    @abstractmethod
    def query(
        self,
        query_texts: list[str],
        n_results: int,
    ) -> QueryResult:
        """Return the n_results closest documents for each query text."""
        ...


# ── AIClient implementations ──────────────────────────────────────────────────

class OpenAIChatClient(AIClient):
    """AIClient backed by the OpenAI SDK."""

    def __init__(self, client: OpenAI) -> None:
        super().__init__()
        self._client = client

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._client.close()

    def parse_jd(self, text: str) -> dict:
        return self.complete_json(
            messages=[
                {"role": "system", "content": self.get_system_prompt_jd_extraction()},
                {"role": "user", "content": text},
            ]
        )

    def complete_json(self, messages: list[dict[str, str]], model: str | None = None) -> dict:
        if not any(m["role"] == "system" for m in messages):
            messages = [
                {"role": "system", "content": self.get_system_prompt_resume_generation()}
            ] + list(messages)
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)


class GoogleAIClient(AIClient):
    """AIClient backed by the Google Gen AI SDK (google-genai)."""

    def __init__(self, client: "_genai_client.Client") -> None:
        super().__init__()
        self._client = client

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._client.close()

    def parse_jd(self, text: str) -> dict:
        return self.complete_json(
            messages=[
                {"role": "system", "content": self.get_system_prompt_jd_extraction()},
                {"role": "user", "content": text},
            ]
        )

    def complete_json(self, messages: list[dict[str, str]], model: str | None = None) -> dict:
        from google.genai import types

        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        if system is None:
            system = self.get_system_prompt_resume_generation()

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction=system,
        )
        response = self._client.models.generate_content(
            model=model or self._model,
            contents=contents,
            config=config,
        )
        return json.loads(response.text)


class AnthropicAIClient(AIClient):
    """AIClient backed by the Anthropic SDK."""

    def __init__(self, client: "_anthropic.Anthropic") -> None:
        super().__init__()
        self._client = client

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._client.close()

    def parse_jd(self, text: str) -> dict:
        return self.complete_json(
            messages=[
                {"role": "system", "content": self.get_system_prompt_jd_extraction()},
                {"role": "user", "content": text},
            ]
        )

    def complete_json(self, messages: list[dict[str, str]], model: str | None = None) -> dict:
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        if system is None:
            system = self.get_system_prompt_resume_generation()

        user_messages = [m for m in messages if m["role"] != "system"]
        response = self._client.messages.create(
            model=model or self._model,
            max_tokens=4096,
            system=system,
            messages=user_messages,
        )
        return json.loads(response.content[0].text)


# ── BulletStore implementations ───────────────────────────────────────────────

class ChromaBulletStore(BulletStore):
    """BulletStore backed by a ChromaDB HTTP collection."""

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass  # ChromaDB HTTP client has no explicit close.

    @staticmethod
    def http_client(connect_with: Union[HostPort, ConnectionString]) -> Any:
        if not isinstance(connect_with, tuple):
            raise TypeError(
                f"ChromaBulletStore requires a HostPort tuple, got {type(connect_with).__name__}. "
                "Set both RESUME_DB_HOST and RESUME_DB_PORT."
            )
        import chromadb
        host, port = connect_with
        return chromadb.HttpClient(host=host, port=int(port))

    def get_or_create_collection(self, client: Any, name: str) -> Any:
        return client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
        )

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_texts: list[str], n_results: int) -> QueryResult:
        return self._collection.query(query_texts=query_texts, n_results=n_results)


class _PgVectorCollection:
    """Internal container pairing a psycopg2 connection with a table and embedding callable."""

    def __init__(self, conn: Any, table_name: str, embedding_fn: Any) -> None:
        self.conn = conn
        self.table_name = table_name
        self.embedding_fn = embedding_fn


class PgVectorBulletStore(BulletStore):
    """BulletStore backed by PostgreSQL with the pgvector extension.

    RESUME_DB_HOST must be a full PostgreSQL DSN:
        postgresql://user:password@localhost:5432/resume_db
    RESUME_DB_PORT must be unset (omit it) so __init__ passes the DSN as a
    ConnectionString rather than a HostPort tuple.
    Embeddings are generated via the ChromaDB OpenAI embedding function using
    RESUME_AI_MODEL_NAME regardless of RESUME_AI_PLATFORM.
    """

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._client.close()

    @staticmethod
    def http_client(connect_with: Union[HostPort, ConnectionString]) -> Any:
        if not isinstance(connect_with, str):
            raise TypeError(
                f"PgVectorBulletStore requires a ConnectionString DSN, "
                f"got {type(connect_with).__name__}. "
                "Set RESUME_DB_HOST to a full PostgreSQL DSN and unset RESUME_DB_PORT."
            )
        import psycopg2
        from pgvector.psycopg2 import register_vector
        conn = psycopg2.connect(connect_with)
        register_vector(conn)
        return conn

    def get_or_create_collection(self, client: Any, name: str) -> Any:
        dimension = len(self.embedding_fn(["probe"])[0])
        with client.cursor() as cur:
            cur.execute(f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {name} (
                    id TEXT PRIMARY KEY,
                    document TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({dimension})
                )
            """)
        client.commit()
        return _PgVectorCollection(client, name, self.embedding_fn)

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        embeddings = self._collection.embedding_fn(documents)
        with self._collection.conn.cursor() as cur:
            for id_, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {self._collection.table_name}
                        (id, document, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (id_, doc, json.dumps(meta), emb),
                )
        self._collection.conn.commit()

    def query(self, query_texts: list[str], n_results: int) -> QueryResult:
        query_embedding = self._collection.embedding_fn(query_texts)[0]
        with self._collection.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, document, metadata
                FROM {self._collection.table_name}
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (query_embedding, n_results),
            )
            rows = cur.fetchall()

        return QueryResult(
            ids=[[row[0] for row in rows]],
            documents=[[row[1] for row in rows]],
            metadatas=[[row[2] for row in rows]],
        )


# ── VTable ────────────────────────────────────────────────────────────────────

class VTable[T]:
    """Registry that maps string keys to factory callables for an ABC."""

    def __init__(self, base: Type[T]) -> None:
        self._base = base
        self._table: dict[str, Callable[[], T]] = {}

    def register(self, key: str, factory: Callable[[], T]) -> None:
        self._table[key] = factory

    def get(self, key: str) -> T:
        factory = self._table.get(key)
        if factory is None:
            valid = ", ".join(self._table)
            raise Exception(
                f"Unknown key '{key}' for {self._base.__name__}. "
                f"Valid choices: {valid}."
            )
        return factory()

    def keys(self) -> tuple[str, ...]:
        return tuple(self._table)


# ── Providers ─────────────────────────────────────────────────────────────────

class AIClientProvider:
    """Reads RESUME_AI_PLATFORM and RESUME_AI_API_KEY and returns the matching AIClient."""

    def __init__(self) -> None:
        def _make_openai() -> OpenAIChatClient:
            return OpenAIChatClient(OpenAI())

        def _make_google() -> GoogleAIClient:
            from google import genai
            return GoogleAIClient(genai.Client())

        def _make_anthropic() -> AnthropicAIClient:
            import anthropic
            return AnthropicAIClient(anthropic.Anthropic())

        self._vtable: VTable[AIClient] = VTable(AIClient)
        self._vtable.register("openai", _make_openai)
        self._vtable.register("google", _make_google)
        self._vtable.register("anthropic", _make_anthropic)

    def get(self) -> AIClient:
        platform = os.getenv("RESUME_AI_PLATFORM")
        if not platform:
            raise Exception("RESUME_AI_PLATFORM must be defined in the script's .env file")
        return self._vtable.get(platform)


class BulletStoreProvider:
    """Reads RESUME_DB_TYPE and returns the matching BulletStore."""

    def __init__(self) -> None:
        self._vtable: VTable[BulletStore] = VTable(BulletStore)
        self._vtable.register("chroma", ChromaBulletStore)
        self._vtable.register("pgvector", PgVectorBulletStore)

    def get(self) -> BulletStore:
        db_type = os.getenv("RESUME_DB_TYPE")
        if not db_type:
            raise Exception("RESUME_DB_TYPE must be defined in the script's .env file")
        return self._vtable.get(db_type)
