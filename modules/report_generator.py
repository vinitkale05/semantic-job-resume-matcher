"""Analysis report serialization helpers."""
import json
from pathlib import Path
from typing import Any


def build_report(match_score: float, resume_skills: list[str], job_skills: list[str], skill_analysis: dict, ai_analysis: Any = None) -> dict:
    return {
        "semantic_match_score": round(float(match_score), 2),
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "skill_analysis": skill_analysis,
        "generative_ai_analysis": ai_analysis,
        "score_note": "Similarity indicator only; not a probability of hiring.",
    }


def save_report(report: dict, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target
