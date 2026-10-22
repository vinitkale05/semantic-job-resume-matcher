from modules.skill_extractor import extract_skills, compare_skills

def test_extract_skills_aliases():
    skills = extract_skills("Built ML models in Python with sklearn and used REST APIs.")
    assert "Python" in skills
    assert "Machine Learning" in skills
    assert "Scikit-learn" in skills
    assert "REST APIs" in skills

def test_compare_skills():
    result = compare_skills(["Python", "SQL"], ["Python", "Flask"])
    assert result["matching_skills"] == ["Python"]
    assert result["missing_skills"] == ["Flask"]
