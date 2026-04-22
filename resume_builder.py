import json
import os
import traceback
from typing import List

from dotenv import load_dotenv

from protocols import AIClient, AIClientProvider, BulletStore, BulletStoreProvider


def load_collection(store: BulletStore, path="data/resume_data.json"):
    with open(path, encoding='utf8') as f:
        data = json.load(f)

    candidate = data["candidate"]
    roles = candidate.get("roles", [])

    role_lookup = {
        (r["company"], r["title"]): r
        for r in roles
    }

    documents = []
    metadatas = []
    ids = []

    for resume in data["resumes"]:
        resume_id = resume["resume_id"]

        for bullet in resume.get("bullets", []):
            text = bullet.get("text")
            if not text:
                continue

            company = bullet.get("company")
            title = bullet.get("title")

            role = role_lookup.get((company, title), {})

            documents.append(text)

            skills = bullet.get("skills", [])
            skills = ", ".join(skills)

            metadatas.append({
                "candidate_name": candidate["name"],
                "resume_id": resume_id,
                "company": company,
                "title": title,
                "dates": role.get("dates", ""),
                "skills": skills,
                "confidence": bullet.get("confidence", "neutral"),
                "focus": resume.get("focus", "")
            })

            ids.append(bullet["id"])

    try:
        store.add(ids=ids, documents=documents, metadatas=metadatas)
    except Exception as e:
        print(f"Error occured: {e}")
        traceback.print_exc()


def retrieve_relevant_bullets(skills: List[str], store: BulletStore, k=20):
    query = " ".join(skills)
    results = store.query(query_texts=[query], n_results=k)
    print("Relevant Bullets retrieved")
    return results["documents"][0]


def generate_bullets_and_skills(
    job_requirements: dict, bullets: List[str], ai: AIClient
):
    user_content = (
        f"Job Requirements:\n{json.dumps(job_requirements, indent=2)}\n\n"
        f"Source Bullets:\n" + "\n".join(f"- {b}" for b in bullets)
    )

    answer = ai.complete_json(
        messages=[{"role": "user", "content": user_content}],
        model="gpt-4.1",
    )
    print("New Bullets, Skills, and Summary Generated")
    return answer['rewritten_bullets'], answer['targeted_skills'], answer['professional_summary']


def match_bullets_to_roles(aligned_bullets, store: BulletStore):
    matched = []

    for text in aligned_bullets:
        result = store.query(query_texts=[text], n_results=1)

        original_id = result["ids"][0][0]
        metadata = result["metadatas"][0][0]

        matched.append({
            "rewritten_text": text,
            "original_bullet_id": original_id,
            "title": metadata["title"],
            "company": metadata["company"],
            "dates": metadata["dates"]
        })

    roles = {}

    for entry in matched:
        title = entry['title']

        if title not in roles:
            roles[title] = {
                "company": entry['company'],
                "title": title,
                "dates": entry['dates'],
                "experiences": []
            }

        roles[title]["experiences"].append(entry['rewritten_text'])

    return roles


def load_static_data(path="data/resume_data.json"):
    with open(path, encoding='utf8') as f:
        data = json.load(f)

    candidate = {}
    candidate['name'] = data['candidate']['name']
    candidate['location'] = data['candidate']['base_location']
    candidate['education'] = data['candidate']['education']
    candidate['portfolio'] = data['candidate']['portfolio_links']
    candidate['certifications'] = data['candidate']['certifications']

    return candidate


def load_experiences(ai: AIClient, store: BulletStore):
    if os.path.exists("data/aligned_experiences.json"):
        with open("data/aligned_experiences.json", 'r') as f:
            saved_data = json.load(f)
        print("Previous experiences loaded")
        return saved_data['experience'], saved_data['targeted_skills'], saved_data['professional_summary']

    print("Loading collection")
    load_collection(store)
    print("Collection loaded")

    with open("data/job_description.txt") as f:
        jd_text = f.read()
    job_req = ai.parse_jd(jd_text)
    print("Job Description Parsed")

    bullets = retrieve_relevant_bullets(job_req["required_skills"], store)

    aligned_bullets, skills, summary = generate_bullets_and_skills(job_req, bullets, ai)

    experience = match_bullets_to_roles(aligned_bullets, store)

    save_data = {
        'professional_summary': summary,
        'experience': experience,
        'targeted_skills': skills,
    }
    with open("data/aligned_experiences.json", 'w') as f:
        json.dump(save_data, f, indent=4)

    return experience, skills, summary


def index_resume_data(path="data/resume_data.json"):
    role_index = {}
    seen = {}

    with open(path, encoding='utf8') as f:
        data = json.load(f)

    print("Indexing Role Data")
    for role in data.get("candidate", {}).get("roles", []):
        title = role.get("title")
        role_index[title] = []
        seen[title] = set()

    for resume in data.get("resumes"):
        for bullet in resume.get("bullets"):
            title = bullet.get("title")
            text = bullet.get("text")
            skills = bullet.get("skills") or []

            if text in seen[title]:
                continue

            role_index[title].append({
                "text": text,
                "skills": [s for s in skills]
            })
            seen[title].add(text)

    return role_index


def pad_roles(
    experience, role_index, min_roles=3, min_bullets=4, path="data/resume_data.json"
):
    with open(path, encoding='utf8') as f:
        resume_data = json.load(f)

    roles_sorted = sorted(
        resume_data["candidate"]["roles"],
        key=lambda r: r.get("start", ""),
        reverse=True
    )

    if len(experience) < min_roles:
        for role in roles_sorted:
            title = role["title"]
            if title not in experience:
                experience[title] = {
                    "company": role.get("company"),
                    "title": title,
                    "dates": role.get("dates"),
                    "experiences": []
                }
            if len(experience) >= min_roles:
                break

    for role_title, role_block in experience.items():
        role_block.setdefault("experiences", [])

        bullets = role_block["experiences"]
        used_text = set(bullets)
        used_skills = set()

        for candidate in role_index.get(role_title, []):
            if len(bullets) >= min_bullets:
                break

            text = candidate["text"]
            skills = set(candidate.get("skills", []))

            if text in used_text:
                continue

            if skills and (skills & used_skills):
                continue

            bullets.append(text)
            used_text.add(text)
            used_skills |= skills

    return experience


if __name__ == "__main__":
    load_dotenv()

    for key, value in os.environ.items():
        if "API_KEY" in key:
            os.environ[f"CHROMA_{key}"] = value

    with AIClientProvider().get() as ai, BulletStoreProvider().get() as store:
        resume = load_static_data()
        role_index = index_resume_data()

        experience, skills, summary = load_experiences(ai, store)
        experience = pad_roles(experience, role_index)

        resume['experiences'] = experience
        resume['skills'] = skills
        resume['professional_summary'] = summary

        with open("data/new_resume.json", 'w') as f:
            json.dump(resume, f, indent=4)

        print("New Resume as JSON \n")
        print(json.dumps(resume, indent=3))
