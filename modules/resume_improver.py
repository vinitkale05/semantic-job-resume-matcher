"""Factual, optional LLM-assisted resume bullet improvement."""
from modules.genai_analyzer import _generate


def improve_bullets(bullets: list[str], job_description: str, api_key: str | None = None) -> str:
    if not bullets:
        raise ValueError("Provide at least one resume bullet point.")
    prompt = f"""Improve the clarity and action-oriented wording of these resume bullets for the target job. Preserve all facts. Do not invent metrics, responsibilities, tools, or achievements. If a claim is unsupported, do not add it. Return each original and proposed rewrite clearly.\n\nBULLETS:\n{chr(10).join('- ' + b for b in bullets)}\n\nJOB DESCRIPTION:\n{job_description[:8000]}"""
    return _generate(prompt, api_key=api_key)
