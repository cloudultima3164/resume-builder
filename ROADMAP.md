# Resume Builder — Development Roadmap

## Overview

This roadmap extends the current single-file, OpenAI-only resume builder into a modular, provider-agnostic system with structured data, versioning, interactive review, and cover letter support. Features are grouped into four phases ordered by dependency: foundational abstractions must land before the flows that consume them.

---

## Phase 1 — Foundation: Abstractions, Data Model & Serialization

> **Goal:** Decouple the tool from any specific AI provider, vector database, or cloud storage backend, and replace raw JSON blobs with a typed, config-driven data model.

### 1.1 — Abstract AI Interface

Define a `BaseAIClient` abstract base class (ABC) that standardizes the operations the rest of the tool requires from any AI provider. All provider-specific logic lives only in the concrete implementations; the calling code never imports a provider SDK directly.

**Interface surface:**
- `chat_completion(messages, response_format, model, **kwargs) → str`
- `embed(texts: list[str], model) → list[list[float]]`

**Implementations:**
- `OpenAIClient` — wraps the existing `openai` SDK (current behavior)
- `AnthropicClient` — wraps `anthropic` SDK; maps the interface to the Messages API
- `GeminiClient` — wraps `google-genai` SDK; maps to `GenerativeModel.generate_content`

**Config:** The active provider and model names are selected via environment variables (e.g., `AI_PROVIDER=anthropic`, `AI_EMBED_MODEL=...`). A factory function reads the config and returns the correct concrete instance.

---

### 1.2 — Abstract Vector Database Interface

Define a `BaseVectorStore` ABC covering the operations used for embedding storage and retrieval.

**Interface surface:**
- `get_or_create_collection(name) → Collection`
- `add(ids, documents, metadatas)`
- `query(query_texts, n_results) → QueryResult`
- `delete(ids)`

**Implementations:**
- `ChromaDBStore` — wraps the existing `chromadb.HttpClient` (current behavior)
- `PgVectorStore` — wraps `psycopg2` / `pgvector` extension; implements the same surface using SQL under the hood

**Config:** `DB_PROVIDER=chroma|pgvector` plus connection parameters read from `.env`.

---

### 1.3 — Abstract Cloud Storage Interface

Define a `BaseStorageClient` ABC for retrieving and persisting resumes (and cover letters) from/to a remote provider.

**Interface surface:**
- `fetch(remote_path) → bytes`
- `save(content: bytes, remote_path: str) → str` (returns a shareable URL or identifier)

**Implementation:**
- `GoogleDriveClient` — wraps the Google Drive API via `google-api-python-client`; authenticates with a service account or OAuth token stored in `.env`

**Config:** `STORAGE_PROVIDER=gdrive` plus credential path / token.

---

### 1.4 — Resume Dataclasses & Mapping Configuration

Replace the stringly-typed JSON handling with native Python dataclasses so that every part of the pipeline operates on well-typed objects.

**Core dataclasses (illustrative, not exhaustive):**

```
Candidate
├── name, location, portfolio_links, certifications
├── education: list[Education]
└── roles: list[Role]

Role
└── company, title, dates, start

Bullet
├── id, text, confidence, skills: list[str], focus
└── version_history: list[BulletVersion]   ← see Phase 3

Resume
├── resume_id, focus
└── bullets: list[Bullet]

CoverLetter                                ← see Phase 2
├── cover_letter_id, target_jd_id
└── paragraphs: list[CoverLetterParagraph]
```

**Mapping configuration** is a YAML or TOML file that describes how fields in the source JSON (or any future source format) map to these dataclasses. Example:

```yaml
resume_data:
  candidate:
    name: candidate.name
    base_location: candidate.location
    roles:
      - company: role.company
        title: role.title
        dates: role.dates
```

This config is what drives the deserializer (§1.5) and is also what the AI mapping-analysis flow (§2.1) produces and refines.

---

### 1.5 — Deserializer: Source Format → Dataclasses

A `ResumeDeserializer` reads the mapping config and uses it to parse the source file (initially `resume_data.json`) into the dataclass tree. This replaces all `data["candidate"]["name"]`-style direct dict access throughout the codebase.

- Validates required fields and raises typed errors on schema violations
- Supports extension to new source formats (DOCX, plain text) by swapping the reader while keeping the same mapping config

---

### 1.6 — Serializer: Dataclasses → Original Format

A `ResumeSerializer` inverts the deserializer: given a dataclass tree, it produces the original JSON structure (or any other target format). This is used when saving the modified resume back to disk or cloud storage.

- Preserves fields not covered by the mapping config (round-trip fidelity)
- Also drives the HTML/PDF render pipeline, replacing the current `render_resume.py` hard-coded access patterns

---

## Phase 2 — Data Ingestion, Embedding Strategy & Job Description Retrieval

> **Goal:** Improve the quality and provenance of data entering the system, and replace manual file-based inputs with automated retrieval.

### 2.1 — AI-Assisted Mapping Configuration Analysis Flow

A standalone script / CLI subcommand (`analyze-mapping`) that:
1. Accepts a raw resume file (JSON, eventually DOCX or plain text)
2. Sends the schema/structure to the AI with a prompt asking it to suggest a mapping config
3. Outputs a candidate YAML mapping config for the user to review, edit, and save

This removes the manual burden of writing the mapping config from scratch and makes onboarding a new resume format a guided, AI-assisted process.

---

### 2.2 — HTTP Client for Job Description Retrieval

An `HttpJobDescriptionClient` that fetches a JD from a URL instead of requiring a local `.txt` file (listed on the existing project wishlist).

- Accepts a URL via CLI argument or config
- Performs the HTTP GET, strips HTML/boilerplate, and returns clean text
- Falls back gracefully to the local file if a URL is not provided
- User-agent and request headers are configurable

---

### 2.3 — Embedding Strategy Review & Extension

The current implementation embeds raw bullet text and attaches only a small set of metadata fields. This milestone audits and extends what enters the vector store:

**Fields to reconsider for embedding / metadata:**
- Structured skill tags (currently comma-joined strings → normalize to structured metadata)
- Confidence scores (already present, but not used in retrieval ranking)
- Role seniority / date range (enables recency weighting)
- JD-specific tags (which jobs was this bullet written for? → see versioning in Phase 3)
- Document type tag: `resume` vs `cover_letter` (see Phase 2.4)
- Source document ID (links back to the versioned element)

**Changes:**
- Expand the `metadatas` schema in both `ChromaDBStore` and `PgVectorStore`
- Update `retrieve_relevant_bullets` to accept filter expressions targeting the richer metadata
- Add cosine-similarity threshold filtering so low-relevance bullets are excluded rather than included

---

### 2.4 — Context Document Ingestion

An optional context document (e.g., a personal bio, LinkedIn export, notes on career goals, or a previous cover letter) can be provided alongside the resume data to give the AI richer background when generating content. This is distinct from the resume source file — it is supplementary prose that may not map to any dataclass field, but informs tone, framing, and emphasis.

**Ingestion:**
- Accepted via a CLI argument (`--context <path>`) or a config entry pointing to a local file or cloud storage path
- The document is read and passed as additional context in the system or user message of the generation prompt, with a clear delimiter so the AI knows its role
- No parsing or field extraction is performed; the document is treated as free-form text

**Embedding considerations:**
- The context document should generally **not** be chunked and embedded the same way as bullets — its value is holistic, not retrievable at the sentence level
- Two viable approaches: (1) inject the full document text into the prompt on every generation call (suitable for shorter documents); (2) for longer documents, chunk by paragraph, embed each chunk, and retrieve only the top-k most relevant chunks relative to the current JD before injecting them — this avoids bloating the prompt for large inputs
- A config option (`CONTEXT_EMBED_STRATEGY=full|chunked`) selects the strategy; `full` is the default
- If chunked, chunks are stored in the vector store under a dedicated collection (e.g., `context_chunks`) with a `source: context` metadata tag so they are never mixed with resume bullet retrieval

---

### 2.5 — Cover Letter Data Model & Embedding Support

Extend the dataclasses, serializer/deserializer, and vector store metadata to accommodate cover letters as first-class entities (full flow is in Phase 4).

---

## Phase 3 — Versioning, Autonomous Mode & Core Flow Hardening

> **Goal:** Give every mutable resume element a traceable history, add a fully automated run mode, and harden error handling throughout.

### 3.1 — Element-Level Versioning

Each leaf element of the resume (individual bullet points, summary paragraphs, skills lists, cover letter paragraphs) carries its own version history. Versions are immutable records that accumulate as the AI modifies content over time.

**`BulletVersion` dataclass:**

```python
@dataclass
class BulletVersion:
    version_id: str        # UUID
    parent_id: str         # ID of the Bullet this version belongs to
    text: str              # The content at this version
    created_at: datetime
    tags: list[str]        # e.g. ["jd:senior-eng-stripe-2025", "ai-rewrite", "user-approved"]
    source_jd_id: str      # FK to the JD that triggered this change
    author: str            # "ai" | "user"
```

- Tags allow both the user and the AI to query history by JD, by author, or by outcome
- The current "active" version for each element is a pointer into the version list, so rollback is trivial
- The serializer writes only the active version's content; the full history is persisted to the vector store metadata and optionally to a local `.versions.json` sidecar

---

### 3.2 — No-Review Flag (`--no-review`)

A CLI flag that bypasses all user confirmation prompts and allows the AI to write changes directly to the dataclass tree, persist them to the vector store, and save the output resume without any human-in-the-loop step.

- Mutually exclusive with interactive review mode (§4.1)
- Each change is still versioned and tagged with `"author": "ai"` and `"auto-approved": true`
- Useful for CI/CD pipelines or batch generation across many JDs

---

## Phase 4 — Interactive Review Mode & Cover Letter Flow

> **Goal:** Add a human-in-the-loop review session for AI suggestions, and extend the entire pipeline to produce cover letters.

### 4.1 — Interactive Review Mode

When run without `--no-review`, the tool enters an interactive session after AI suggestions are generated but before they are committed. The session is a turn-based dialogue between the user and the AI.

**Flow:**
1. AI presents a proposed change (e.g., a rewritten bullet)
2. User may: `accept`, `reject`, `edit <new text>`, or `ask <question>` to have the AI iterate
3. On `ask`, the AI receives the full conversation history and the current draft, responds, and presents a revised suggestion
4. Accepted changes are written to the dataclass tree, versioned with `"author": "user-approved"`, embedded, and saved
5. Rejected changes are discarded; the original version is retained
6. Session state is checkpointed after each decision so the session can be resumed if interrupted

---

### 4.2 — Full Cover Letter Flow

Parallel to the resume flow, a cover letter pipeline that reuses all the same abstractions:

1. **Retrieval:** Query the vector store for bullets and past cover letter paragraphs relevant to the target JD
2. **Generation:** AI drafts a full cover letter using the JD, candidate summary, and retrieved content
3. **Review / No-review:** Same interactive or autonomous mode as the resume flow
4. **Versioning:** Each paragraph is versioned with JD tags, identical to bullet versioning
5. **Storage:** Serialized and saved via the cloud storage interface alongside the resume
6. **Rendering:** A new template in `templates/` renders the cover letter to HTML (and optionally PDF)

---

## Dependency Graph

```
Phase 1 (all items)
      │
      ├──► Phase 2.1 (mapping analysis — needs AI interface + dataclasses)
      ├──► Phase 2.2 (HTTP client — standalone, no phase 1 hard dep)
      ├──► Phase 2.3 (embedding strategy — needs DB interface + dataclasses)
      ├──► Phase 2.4 (context document ingestion — needs AI interface + DB interface)
      └──► Phase 2.5 (cover letter model — needs dataclasses)
                │
                └──► Phase 3 (versioning + no-review — needs all of Phase 1 & 2)
                            │
                            └──► Phase 4 (interactive review + cover letter flow)
```

---

## Suggested File / Module Layout (Post-Refactor)

```
resume-builder/
├── config/
│   └── mapping.yaml              # Resume field mapping config
├── data/
│   ├── resume_data.json
│   └── *.versions.json           # Per-element version sidecars
├── output/
├── templates/
├── resume_builder/               # Main package
│   ├── __init__.py
│   ├── ai/
│   │   ├── base.py               # BaseAIClient ABC
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   └── gemini_client.py
│   ├── db/
│   │   ├── base.py               # BaseVectorStore ABC
│   │   ├── chroma_store.py
│   │   └── pgvector_store.py
│   ├── storage/
│   │   ├── base.py               # BaseStorageClient ABC
│   │   └── gdrive_client.py
│   ├── models/
│   │   ├── resume.py             # Dataclasses
│   │   ├── cover_letter.py
│   │   └── versioning.py
│   ├── io/
│   │   ├── deserializer.py
│   │   ├── serializer.py
│   │   ├── http_jd_client.py
│   │   └── mapping_analyzer.py  # AI-assisted mapping flow
│   ├── flow/
│   │   ├── resume_flow.py        # Core resume pipeline
│   │   ├── cover_letter_flow.py
│   │   └── review_session.py    # Interactive review loop
│   └── prompts.py
├── .env.sample
├── requirements.txt
└── README.md
```


