"""Rule-based skill extraction with aliases; extend SKILL_ALIASES as needed."""
import re

SKILL_ALIASES = {
    "Python": ["python"], "Java": ["java"], "C++": ["c++", "cpp"],
    "JavaScript": ["javascript", "js"], "SQL": ["sql", "structured query language"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"], "Generative AI": ["generative ai", "genai", "gen ai"],
    "Natural Language Processing": ["natural language processing", "nlp"],
    "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"],
    "Scikit-learn": ["scikit-learn", "sklearn"], "Pandas": ["pandas"], "NumPy": ["numpy"],
    "React": ["react", "react.js"], "Node.js": ["node.js", "nodejs"],
    "Flask": ["flask"], "Django": ["django"], "REST APIs": ["rest api", "rest apis", "restful api", "restful apis"],
    "Git": ["git", "github"], "Docker": ["docker"], "Cloud Computing": ["cloud computing", "aws", "azure", "gcp"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Data Visualization": ["data visualization", "data visualisation", "plotly", "tableau", "power bi"],
    "Large Language Models": ["large language model", "large language models", "llm", "llms"],
    "Retrieval-Augmented Generation": ["retrieval-augmented generation", "rag"],
    "Vector Databases": ["vector database", "vector databases", "faiss", "chroma"],
    "Prompt Engineering": ["prompt engineering"], "MySQL": ["mysql"], "SQLite": ["sqlite"],
    "HTML": ["html"], "CSS": ["css"], "FastAPI": ["fastapi"],
}


def extract_skills(text: str) -> list[str]:
    found = []
    normalized = text.lower()
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            a = alias.lower()
            if a == "c++":
                pattern = r"(?<![a-z0-9])c\+\+(?![a-z0-9+])"
            elif a == "js":
                pattern = r"(?<![a-z0-9.])js(?![a-z0-9])"
            elif "-" in a or "." in a:
                pattern = r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])"
            else:
                pattern = r"\b" + re.escape(a) + r"\b"
            if re.search(pattern, normalized):
                found.append(canonical)
                break
    return sorted(set(found))


def compare_skills(resume_skills, job_skills) -> dict[str, list[str]]:
    resume_set, job_set = set(resume_skills), set(job_skills)
    return {
        "matching_skills": sorted(resume_set & job_set),
        "missing_skills": sorted(job_set - resume_set),
        "resume_skills_not_in_job_description": sorted(resume_set - job_set),
        "job_skills_identified": sorted(job_set),
        "resume_skills_identified": sorted(resume_set),
    }
