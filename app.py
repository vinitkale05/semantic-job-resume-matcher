"""Streamlit app for Semantic Job–Resume Matcher Using Generative AI.
A comprehensive B.Tech Computer Science Generative AI Capstone Project.
"""
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.resume_parser import parse_resume
from modules.semantic_matcher import (
    embed_chunks,
    chunk_similarity,
    similarity_score_to_percent,
    explain_chunk_matches,
)
from modules.skill_extractor import extract_skills, compare_skills
from modules.genai_analyzer import analyse_match, _is_configured
from modules.resume_improver import improve_bullets
from modules.validators import validate_upload, validate_text
from modules.report_generator import build_report

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Semantic Job-Resume Matcher | GenAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Modern Styling System (Dark Navy, Subtle Blue, Slate)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Banner */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 26px 30px;
    color: #f8fafc;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #ffffff 0%, #93c5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #cbd5e1;
    margin-bottom: 12px;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(96, 165, 250, 0.4);
    color: #93c5fd;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(15, 23, 42, 0.08);
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
    margin-bottom: 6px;
}

.metric-caption {
    font-size: 0.82rem;
    color: #94a3b8;
}

/* Skill Tags */
.skill-tag {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}

.skill-match {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #bbf7d0;
}

.skill-missing {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fecaca;
}

.skill-extra {
    background-color: #e0f2fe;
    color: #075985;
    border: 1px solid #bae6fd;
}

/* Evidence Card */
.evidence-card {
    background: #f8fafc;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}

.evidence-header {
    display: flex;
    justify-content: space-between;
    font-weight: 700;
    font-size: 0.9rem;
    color: #1e293b;
    margin-bottom: 8px;
}

.score-badge {
    background: #3b82f6;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
}

.upload-success-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.85rem;
    color: #166534;
    margin-top: 8px;
    margin-bottom: 8px;
}

.disclaimer-box {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.83rem;
    color: #64748b;
    margin-bottom: 18px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_text" not in st.session_state:
    st.session_state.job_text = ""
if "resume_file_name" not in st.session_state:
    st.session_state.resume_file_name = None
if "job_file_name" not in st.session_state:
    st.session_state.job_file_name = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "api_key_input" not in st.session_state:
    st.session_state.api_key_input = ""


def extract_uploaded_resume(file_obj) -> str:
    """Extract text from uploaded resume (PDF, DOCX, or TXT)."""
    if file_obj is None:
        return ""
    name = getattr(file_obj, "name", "").lower()
    data = file_obj.getvalue()
    if name.endswith(".pdf") or name.endswith(".docx"):
        return parse_resume(file_obj)
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()
    else:
        raise ValueError("Unsupported resume format. Please upload PDF, DOCX, or TXT.")


def extract_uploaded_job(file_obj) -> str:
    """Extract text from uploaded job description (TXT, PDF, or DOCX)."""
    if file_obj is None:
        return ""
    name = getattr(file_obj, "name", "").lower()
    data = file_obj.getvalue()
    if name.endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text("text") for page in pdf).strip()
    elif name.endswith(".docx"):
        from docx import Document
        doc = Document(BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(c.text.strip() for c in row.cells))
        return "\n".join(paragraphs).strip()
    else:
        # Default text file
        return data.decode("utf-8", errors="replace").strip()


def load_sample_inputs():
    """Load sample resume and job description from sample_data/."""
    sample_dir = Path(__file__).parent / "sample_data"
    resume_path = sample_dir / "sample_resume.txt"
    job_path = sample_dir / "sample_job_description.txt"

    if resume_path.exists():
        st.session_state.resume_text = resume_path.read_text(encoding="utf-8").strip()
        st.session_state.resume_file_name = "sample_resume.txt"
    if job_path.exists():
        st.session_state.job_text = job_path.read_text(encoding="utf-8").strip()
        st.session_state.job_file_name = "sample_job_description.txt"


def clear_inputs():
    """Reset all input fields and analysis results."""
    st.session_state.resume_text = ""
    st.session_state.job_text = ""
    st.session_state.resume_file_name = None
    st.session_state.job_file_name = None
    st.session_state.analysis_results = None


# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Control Center")

    st.markdown(
        """
        <div style="background:#1e293b; color:#93c5fd; padding:10px 14px; border-radius:8px; font-size:0.82rem; margin-bottom:14px; border:1px solid #334155;">
            🎓 <b>B.Tech AI Capstone Project</b><br>
            Vector Embeddings + Cosine Similarity + Generative AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚡ Load Demo", help="Load pre-built sample resume and ML job description", use_container_width=True):
            load_sample_inputs()
            st.rerun()
    with col_btn2:
        if st.button("🗑️ Reset All", help="Clear all fields and results", use_container_width=True):
            clear_inputs()
            st.rerun()

    st.divider()

    st.markdown("#### 🤖 Generative AI (Gemini)")
    gemini_ready = _is_configured(st.session_state.api_key_input)
    if gemini_ready:
        st.success("🟢 Gemini API Configured", icon="✅")
    else:
        st.warning("🟡 Semantic-Only Mode", icon="ℹ️")

    enable_genai = st.checkbox("Enable Generative AI Insights", value=True)

    with st.expander("🔑 API Key Settings"):
        custom_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            value=st.session_state.api_key_input,
            placeholder="AIzaSy...",
            help="Optional: Provide or override Gemini API key for live LLM feedback.",
        )
        if custom_key != st.session_state.api_key_input:
            st.session_state.api_key_input = custom_key
            st.rerun()
        st.caption("Key is retained in local session memory only.")

    st.divider()

    st.markdown("#### 📐 Architecture Spec")
    st.markdown(
        """
        - **Embedding Model:** `all-MiniLM-L6-v2`
        - **Vector Dimension:** 384 dimensions
        - **Distance Metric:** Cosine Similarity
        - **Semantic Chunking:** Boundary split (~900 chars)
        - **LLM Engine:** Gemini 2.5 Flash
        """
    )


# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <span class="hero-badge">AI-Powered Recruitment Analytics</span>
        <div class="hero-title">Semantic Job–Resume Matcher</div>
        <div class="hero-subtitle">
            Beyond keyword matching: Deep semantic sentence embeddings, vector cosine alignment, and grounded Generative AI feedback.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚖️ <b>Academic Ethics & Transparency Note:</b> The calculated similarity score reflects mathematical vector alignment between text embeddings.
        It is an objective semantic relevance metric, not an automated hiring probability. All analysis runs locally; resume data is never retained.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Input Section with Flexible Input Modes
# ---------------------------------------------------------
st.markdown("### 📥 Input Resume & Job Description")

input_mode = st.radio(
    "Select Input Method:",
    options=["📁 Upload Documents (PDF / DOCX / TXT)", "📝 Paste Text Directly", "⚡ Quick Sample Demo"],
    horizontal=True,
)

if input_mode == "📁 Upload Documents (PDF / DOCX / TXT)":
    st.caption("Simply upload your files below. Text is automatically extracted—**no manual text insertion required**.")
    col_up_res, col_up_job = st.columns(2)

    with col_up_res:
        st.markdown("#### 📄 Candidate Resume")
        up_res_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            key="doc_uploader_resume",
        )
        if up_res_file is not None:
            try:
                extracted_res = extract_uploaded_resume(up_res_file)
                st.session_state.resume_text = extracted_res
                st.session_state.resume_file_name = up_res_file.name
                st.markdown(
                    f"""
                    <div class="upload-success-box">
                        ✅ <b>{up_res_file.name}</b>: Successfully extracted {len(extracted_res.split())} words ({len(extracted_res)} characters).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("👁️ View Extracted Resume Text"):
                    st.text_area("Extracted Resume", value=extracted_res, height=180, disabled=True, key="preview_res")
            except Exception as e:
                st.error(f"Failed to extract resume text: {e}")
        elif st.session_state.resume_text:
            st.info(f"Loaded resume: {st.session_state.resume_file_name or 'Custom text'} ({len(st.session_state.resume_text.split())} words)")
            with st.expander("👁️ View Current Resume Text"):
                st.text_area("Current Resume", value=st.session_state.resume_text, height=150, disabled=True, key="curr_res_preview")

    with col_up_job:
        st.markdown("#### 💼 Target Job Description")
        up_job_file = st.file_uploader(
            "Upload Job Description (TXT, PDF, or DOCX)",
            type=["txt", "pdf", "docx"],
            key="doc_uploader_job",
        )
        if up_job_file is not None:
            try:
                extracted_job = extract_uploaded_job(up_job_file)
                st.session_state.job_text = extracted_job
                st.session_state.job_file_name = up_job_file.name
                st.markdown(
                    f"""
                    <div class="upload-success-box">
                        ✅ <b>{up_job_file.name}</b>: Successfully extracted {len(extracted_job.split())} words ({len(extracted_job)} characters).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("👁️ View Extracted Job Description"):
                    st.text_area("Extracted Job Description", value=extracted_job, height=180, disabled=True, key="preview_job")
            except Exception as e:
                st.error(f"Failed to extract job description text: {e}")
        elif st.session_state.job_text:
            st.info(f"Loaded job description: {st.session_state.job_file_name or 'Custom text'} ({len(st.session_state.job_text.split())} words)")
            with st.expander("👁️ View Current Job Text"):
                st.text_area("Current Job", value=st.session_state.job_text, height=150, disabled=True, key="curr_job_preview")

elif input_mode == "📝 Paste Text Directly":
    col_p_res, col_p_job = st.columns(2)
    with col_p_res:
        st.markdown("#### 📄 Candidate Resume")
        res_area = st.text_area(
            "Paste resume text content:",
            value=st.session_state.resume_text,
            height=240,
            placeholder="Paste candidate resume text here...",
            key="paste_resume_textarea",
        )
        if res_area != st.session_state.resume_text:
            st.session_state.resume_text = res_area
            st.session_state.resume_file_name = "Pasted Resume Text"

    with col_p_job:
        st.markdown("#### 💼 Target Job Description")
        job_area = st.text_area(
            "Paste job description content:",
            value=st.session_state.job_text,
            height=240,
            placeholder="Paste target job description and requirements here...",
            key="paste_job_textarea",
        )
        if job_area != st.session_state.job_text:
            st.session_state.job_text = job_area
            st.session_state.job_file_name = "Pasted Job Description"

else:
    # ⚡ Quick Sample Demo
    st.info("Demonstration mode: Preloaded sample Machine Learning Engineer resume and job description.")
    if not st.session_state.resume_text or not st.session_state.job_text:
        load_sample_inputs()

    col_s_res, col_s_job = st.columns(2)
    with col_s_res:
        st.markdown("#### 📄 Sample Resume (Junior AI Developer)")
        st.text_area("Sample Resume", value=st.session_state.resume_text, height=220, disabled=True, key="sample_res_view")
    with col_s_job:
        st.markdown("#### 💼 Sample Job (Junior ML Engineer)")
        st.text_area("Sample Job Description", value=st.session_state.job_text, height=220, disabled=True, key="sample_job_view")


# ---------------------------------------------------------
# Primary Analysis Button
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
run_analysis = st.button("🚀 Analyze Semantic Match & Skill Fit", type="primary", use_container_width=True)

if run_analysis:
    resume_curr = st.session_state.resume_text.strip()
    job_curr = st.session_state.job_text.strip()

    valid_res, res_err = validate_text(resume_curr, "Resume", min_chars=30)
    valid_job, job_err = validate_text(job_curr, "Job description", min_chars=30)

    if not valid_res:
        st.error(f"❌ Resume Error: {res_err} Please upload a valid document or paste text.")
    elif not valid_job:
        st.error(f"❌ Job Description Error: {job_err} Please upload a valid document or paste text.")
    else:
        with st.spinner("🧠 Computing 384-dimensional sentence embeddings and cosine alignment..."):
            try:
                # 1. Semantic Embedding & Similarity
                resume_embeddings = embed_chunks(resume_curr)
                job_embeddings = embed_chunks(job_curr)
                raw_score = chunk_similarity(resume_embeddings, job_embeddings)
                match_score = similarity_score_to_percent(raw_score)
                evidence = explain_chunk_matches(resume_curr, job_curr, top_n=6)

                # 2. Skill Extraction & Overlap
                resume_skills = extract_skills(resume_curr)
                job_skills = extract_skills(job_curr)
                skill_results = compare_skills(resume_skills, job_skills)

                # 3. Optional Generative AI Analysis
                ai_analysis = None
                active_key = st.session_state.api_key_input
                if enable_genai:
                    if _is_configured(active_key):
                        with st.spinner("✨ Synthesizing grounded recommendations via Gemini API..."):
                            try:
                                ai_analysis = analyse_match(
                                    resume_curr,
                                    job_curr,
                                    match_score,
                                    skill_results,
                                    api_key=active_key if active_key else None,
                                )
                            except Exception as gen_err:
                                st.warning(f"Generative AI analysis skipped ({type(gen_err).__name__}: {gen_err}). Semantic scores remain 100% accurate.")
                    else:
                        st.info("💡 Semantic analysis completed. Add your GEMINI_API_KEY in the sidebar or .env file to unlock Generative AI explanations.")

                # Save into session state for persistent rendering
                st.session_state.analysis_results = {
                    "match_score": match_score,
                    "raw_score": raw_score,
                    "evidence": evidence,
                    "resume_skills": resume_skills,
                    "job_skills": job_skills,
                    "skill_results": skill_results,
                    "ai_analysis": ai_analysis,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.success("✅ Analysis completed successfully!")
            except Exception as e:
                st.error(f"Analysis pipeline failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------
# Results Dashboard (Rendered when results exist)
# ---------------------------------------------------------
results = st.session_state.analysis_results

if results is not None:
    st.divider()
    st.markdown("## 📊 Match Analytics & Evaluation Dashboard")

    match_score = results["match_score"]
    resume_skills = results["resume_skills"]
    job_skills = results["job_skills"]
    skill_results = results["skill_results"]
    evidence = results["evidence"]
    ai_analysis = results["ai_analysis"]

    # Calculate skill coverage ratio
    matching_count = len(skill_results["matching_skills"])
    total_job_skills = len(job_skills)
    skill_coverage = (matching_count / total_job_skills * 100.0) if total_job_skills > 0 else 0.0

    # Determine match tier color
    if match_score >= 75:
        tier_label = "Strong Fit"
        score_color = "#10b981"
    elif match_score >= 50:
        tier_label = "Moderate Fit"
        score_color = "#3b82f6"
    else:
        tier_label = "Needs Optimization"
        score_color = "#f59e0b"

    # Metric Row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Semantic Match</div>
                <div class="metric-value" style="color:{score_color};">{match_score:.1f}%</div>
                <div class="metric-caption">Cosine Similarity ({tier_label})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Skill Coverage</div>
                <div class="metric-value">{skill_coverage:.0f}%</div>
                <div class="metric-caption">{matching_count} of {total_job_skills} Required Skills</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Resume Skills Found</div>
                <div class="metric-value">{len(resume_skills)}</div>
                <div class="metric-caption">Across NLP Canonical Lexicon</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Missing Skills</div>
                <div class="metric-value" style="color:#ef4444;">{len(skill_results['missing_skills'])}</div>
                <div class="metric-caption">Unmatched Job Requirements</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 7 All-Inclusive Functional Tabs
    # -----------------------------------------------------
    tab_overview, tab_skills, tab_evidence, tab_genai, tab_bullet, tab_theory, tab_export = st.tabs([
        "📈 Overview & Chart",
        "🧠 Skill Gap Analysis",
        "🔍 Explainable Evidence",
        "✨ Generative AI Insights",
        "✍️ Resume Bullet Rewriter",
        "📚 Academic Theory & Pipeline",
        "💾 Export & Report",
    ])

    # -----------------------------------------------------
    # TAB 1: 📈 Overview & Chart
    # -----------------------------------------------------
    with tab_overview:
        col_c1, col_c2 = st.columns([3, 2])

        with col_c1:
            st.markdown("#### 📊 Skill Distribution Breakdown")
            chart_data = pd.DataFrame([
                {"Category": "Matching Skills", "Count": len(skill_results["matching_skills"])},
                {"Category": "Missing from Resume", "Count": len(skill_results["missing_skills"])},
                {"Category": "Transferable (Resume-only)", "Count": len(skill_results["resume_skills_not_in_job_description"])},
            ])
            fig = px.bar(
                chart_data,
                x="Category",
                y="Count",
                text="Count",
                color="Category",
                color_discrete_map={
                    "Matching Skills": "#10b981",
                    "Missing from Resume": "#ef4444",
                    "Transferable (Resume-only)": "#3b82f6",
                },
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                xaxis_title="",
                yaxis_title="Identified Competencies",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_c2:
            st.markdown("#### 🎯 Match Level Summary")
            st.progress(max(0.0, min(match_score / 100.0, 1.0)))

            st.markdown(
                f"""
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:14px;">
                    <div style="font-weight:700; color:#0f172a; margin-bottom:6px;">Summary Evaluation:</div>
                    <p style="font-size:0.88rem; color:#475569; margin-bottom:10px;">
                        The candidate's resume exhibits <b>{tier_label}</b> ({match_score:.1f}% semantic similarity) with the target job profile.
                        Skill overlap is currently at <b>{skill_coverage:.1f}%</b>.
                    </p>
                    <ul style="font-size:0.85rem; color:#475569; padding-left:20px; margin-bottom:0;">
                        <li><b>{matching_count}</b> required skills matched directly.</li>
                        <li><b>{len(skill_results['missing_skills'])}</b> target skills not detected in resume text.</li>
                        <li><b>{len(skill_results['resume_skills_not_in_job_description'])}</b> additional technical skills present in candidate profile.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # TAB 2: 🧠 Skill Gap Analysis
    # -----------------------------------------------------
    with tab_skills:
        st.markdown("#### 🧠 Detailed Skill Inventory & Overlap")
        st.caption("Pills indicate detected canonical technical competencies extracted using NLP rule-based matching with alias normalization.")

        sk_col1, sk_col2, sk_col3 = st.columns(3)

        with sk_col1:
            st.markdown("##### 🟢 Matching Skills")
            if skill_results["matching_skills"]:
                tags_html = "".join([f'<span class="skill-tag skill-match">✓ {s}</span>' for s in skill_results["matching_skills"]])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.info("No exact matching skills found.")

        with sk_col2:
            st.markdown("##### 🔴 Missing From Resume")
            if skill_results["missing_skills"]:
                tags_html = "".join([f'<span class="skill-tag skill-missing">✗ {s}</span>' for s in skill_results["missing_skills"]])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.caption("Consider adding verified experience in these areas to optimize ATS alignment.")
            else:
                st.success("All requested job skills are present!")

        with sk_col3:
            st.markdown("##### 🔵 Additional Resume Skills")
            if skill_results["resume_skills_not_in_job_description"]:
                tags_html = "".join([f'<span class="skill-tag skill-extra">+ {s}</span>' for s in skill_results["resume_skills_not_in_job_description"]])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.caption("Valuable transferable skills not explicitly listed in the job description.")
            else:
                st.write("No extra skills detected.")

    # -----------------------------------------------------
    # TAB 3: 🔍 Explainable Evidence
    # -----------------------------------------------------
    with tab_evidence:
        st.markdown("#### 🔍 Explainable Semantic Alignment (Chunk-Level Evidence)")
        st.caption(
            "Dense vector embeddings break text into semantically cohesive passages. "
            "Below are the top job requirement chunks paired with the closest corresponding evidence extracted from the candidate's resume."
        )

        for idx, item in enumerate(evidence, 1):
            sim_val = item["similarity"]
            sim_pct = max(0.0, min(sim_val * 100.0, 100.0))

            badge_color = "#10b981" if sim_val >= 0.65 else "#3b82f6" if sim_val >= 0.45 else "#64748b"

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-header">
                        <span>Match #{idx}</span>
                        <span class="score-badge" style="background:{badge_color};">Cosine Similarity: {sim_val:.3f} ({sim_pct:.1f}%)</span>
                    </div>
                    <div style="font-size:0.85rem; color:#475569; margin-bottom:8px;">
                        <b>💼 Job Requirement Passage:</b><br>
                        <span style="color:#0f172a;">"{item['job_requirement_excerpt']}"</span>
                    </div>
                    <div style="font-size:0.85rem; color:#475569;">
                        <b>📄 Closest Resume Evidence:</b><br>
                        <span style="color:#1e40af;">"{item['resume_evidence_excerpt']}"</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------
    # TAB 4: ✨ Generative AI Insights
    # -----------------------------------------------------
    with tab_genai:
        st.markdown("#### ✨ Grounded Generative AI Feedback")

        if ai_analysis and isinstance(ai_analysis, dict):
            if "compatibility_explanation" in ai_analysis:
                st.markdown("##### 📌 Compatibility Assessment")
                st.info(ai_analysis["compatibility_explanation"])

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                if "matching_strengths" in ai_analysis and ai_analysis["matching_strengths"]:
                    st.markdown("##### 💪 Key Candidate Strengths")
                    for s in ai_analysis["matching_strengths"]:
                        st.markdown(f"- {s}")

                if "missing_or_unverified_skills" in ai_analysis and ai_analysis["missing_or_unverified_skills"]:
                    st.markdown("##### ⚠️ Skill Gaps & Verification Needs")
                    for g in ai_analysis["missing_or_unverified_skills"]:
                        st.markdown(f"- {g}")

            with col_g2:
                if "resume_improvement_suggestions" in ai_analysis and ai_analysis["resume_improvement_suggestions"]:
                    st.markdown("##### 💡 Strategic Resume Recommendations")
                    for r in ai_analysis["resume_improvement_suggestions"]:
                        st.markdown(f"- {r}")

                if "learning_recommendations" in ai_analysis and ai_analysis["learning_recommendations"]:
                    st.markdown("##### 📈 30-Day Upskilling Roadmap")
                    for l in ai_analysis["learning_recommendations"]:
                        st.markdown(f"- {l}")

            if "tailored_resume_summary" in ai_analysis:
                st.markdown("##### 📝 Suggested Tailored Resume Summary")
                st.code(ai_analysis["tailored_resume_summary"], language="markdown")

            if "final_assessment" in ai_analysis:
                st.markdown("##### 🏁 Final Verdict")
                st.write(ai_analysis["final_assessment"])

        else:
            st.info(
                """
                **Gemini Generative AI is currently unconfigured or disabled.**
                To activate automated generative feedback:
                1. Set `GEMINI_API_KEY=your_key` in `.env`, or
                2. Enter your key in the **Sidebar ➔ 🔑 API Key Settings**, then re-run analysis.
                """
            )

            # Academic Mock Preview for student presentations & defense
            with st.expander("👁️ View Sample Academic GenAI Output (Preview)"):
                st.markdown(
                    """
                    > **Sample Grounded Response Demonstration:**
                    >
                    > **Compatibility Assessment:** The candidate demonstrates strong foundational qualifications in Python, Machine Learning, and data wrangling, but requires tangible evidence of REST API creation and Docker containerization.
                    >
                    > **Candidate Strengths:**
                    > - Solid algorithmic groundwork using Scikit-learn and Pandas.
                    > - Practical experience building web demos using Flask.
                    >
                    > **Identified Gaps:**
                    > - Production REST APIs (FastAPI/Flask RESTful endpoints).
                    > - Containerization & Docker deployment mentioned in preferred qualifications.
                    >
                    > **Tailored Resume Summary:**
                    > *"Results-driven Junior Machine Learning Developer with strong hands-on expertise in Python, predictive modeling with Scikit-learn, and dataset engineering with Pandas/NumPy. Proven ability to translate business requirements into working prototypes."*
                    """
                )

    # -----------------------------------------------------
    # TAB 5: ✍️ Resume Bullet Rewriter
    # -----------------------------------------------------
    with tab_bullet:
        st.markdown("#### ✍️ AI-Powered Bullet Point Rewriter")
        st.caption(
            "Upgrade your resume bullet points into impact-driven statements following Google's XYZ formula: "
            "**'Accomplished [X] as measured by [Y], by doing [Z]'** without fabricating facts."
        )

        sample_bullet_default = (
            "Built a Python machine learning pipeline with Scikit-learn to classify structured data."
        )
        user_bullet = st.text_area(
            "Enter an existing bullet point from your resume:",
            value=sample_bullet_default,
            height=90,
            key="bullet_rewrite_input",
        )

        if st.button("✨ Generate Optimized Bullet Rewrite", type="primary", key="btn_rewrite"):
            if not user_bullet.strip():
                st.warning("Please provide a bullet point to improve.")
            elif not _is_configured(st.session_state.api_key_input):
                st.warning("Please configure your GEMINI_API_KEY in the sidebar or `.env` file to use the live bullet rewriter.")
                st.markdown(
                    """
                    **Rule-Based Formula Recommendation (Fallback):**
                    > *Original:* "Built a Python machine learning pipeline with Scikit-learn to classify structured data."  
                    > *Recommended Rewrite:* **"Engineered and evaluated an end-to-end classification pipeline in Python using Scikit-learn, standardizing data pre-processing and model validation workflows."**
                    """
                )
            else:
                with st.spinner("Crafting factual, high-impact bullet rewrite..."):
                    try:
                        rewrite_res = improve_bullets(
                            [user_bullet.strip()],
                            st.session_state.job_text,
                            api_key=st.session_state.api_key_input or None,
                        )
                        st.markdown("##### 🎯 Suggested Improvement:")
                        st.write(rewrite_res)
                    except Exception as err:
                        st.error(f"Rewrite error: {err}")

    # -----------------------------------------------------
    # TAB 6: 📚 Academic Theory & Pipeline
    # -----------------------------------------------------
    with tab_theory:
        st.markdown("#### 🎓 Academic Theory & NLP Architecture")

        st.markdown("##### 1. Keyword Matching vs. Semantic Matching")
        comparison_table = pd.DataFrame([
            {
                "Feature": "Core Mechanism",
                "Keyword Matching (Traditional ATS)": "Exact lexical string / token matching (BM25, TF-IDF)",
                "Semantic Matching (Our System)": "Dense vector embeddings in 384-dimensional latent space",
            },
            {
                "Feature": "Synonym Handling",
                "Keyword Matching (Traditional ATS)": "Fails if exact word is missing (e.g., 'ML' vs 'Machine Learning')",
                "Semantic Matching (Our System)": "Recognizes semantic similarity regardless of surface phrasing",
            },
            {
                "Feature": "Contextual Awareness",
                "Keyword Matching (Traditional ATS)": "Ignores word order, syntax, and relational context",
                "Semantic Matching (Our System)": "Captures contextual semantics across sentences",
            },
            {
                "Feature": "Vulnerability to Cheating",
                "Keyword Matching (Traditional ATS)": "Vulnerable to 'keyword stuffing' (hidden white text)",
                "Semantic Matching (Our System)": "Requires coherent passage-level context to yield high vector cosine score",
            },
        ])
        st.table(comparison_table)

        st.markdown("##### 2. Mathematical Definition: Cosine Similarity")
        st.latex(
            r"\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}"
        )
        st.markdown(
            r"""
            - **Normalized Embeddings:** Because embeddings from `all-MiniLM-L6-v2` are $L_2$-normalized ($\|\mathbf{u}\|_2 = 1$), the cosine similarity simplifies directly to the inner dot product: $\mathbf{u} \cdot \mathbf{v}$.
            - **Chunk-Aggregation Algorithm:** The overall match score is computed as the mean of the maximal resume-chunk similarity for each job description chunk:
            """
        )
        st.latex(
            r"\text{Match Score} = \frac{1}{|J|} \sum_{j \in J} \max_{r \in R} \cos(\mathbf{e}_j, \mathbf{e}_r)"
        )

        st.markdown("##### 3. End-to-End Generative AI Pipeline")
        st.markdown(
            """
            ```
            [Resume (PDF/DOCX/Text)]      [Job Description (Text)]
                        │                            │
                        ▼                            ▼
            [Text Preprocessing & Semantic Chunking]
                        │                            │
                        ▼                            ▼
            [Sentence-Transformers (all-MiniLM-L6-v2 Embeddings)]
                        │                            │
                        └─────────────┬──────────────┘
                                      ▼
                        [Scikit-Learn Cosine Similarity Matrix]
                                      │
                        ┌─────────────┴──────────────┐
                        ▼                            ▼
              [Semantic Score & Evidence]     [Rule-Based Skill Extraction]
                        │                            │
                        └─────────────┬──────────────┘
                                      ▼
                      [Grounded Prompt Synthesis]
                                      ▼
                     [Gemini 2.5 Flash Generative AI]
                                      ▼
                [Executive Dashboard & Downloadable JSON Report]
            ```
            """
        )

    # -----------------------------------------------------
    # TAB 7: 💾 Export & Reports
    # -----------------------------------------------------
    with tab_export:
        st.markdown("#### 💾 Export Analysis Report")
        st.caption("Generate complete machine-readable and human-readable audit reports for your records or academic evaluation.")

        report_dict = build_report(
            match_score=match_score,
            resume_skills=resume_skills,
            job_skills=job_skills,
            skill_analysis=skill_results,
            ai_analysis=ai_analysis,
        )
        report_dict["evaluation_timestamp"] = results["timestamp"]
        report_dict["model_spec"] = "sentence-transformers/all-MiniLM-L6-v2"

        json_data = json.dumps(report_dict, indent=2, ensure_ascii=False, default=str)

        # Markdown report generator
        md_report = f"""# Semantic Job-Resume Match Evaluation Report
*Generated on: {results['timestamp']}*

## Executive Summary
- **Overall Semantic Match:** {match_score:.1f}% ({tier_label})
- **Skill Coverage:** {skill_coverage:.1f}% ({matching_count}/{total_job_skills} skills)
- **Resume Skills Detected:** {', '.join(resume_skills) if resume_skills else 'None'}
- **Missing Job Requirements:** {', '.join(skill_results['missing_skills']) if skill_results['missing_skills'] else 'None'}

## Top Semantic Chunk Evidence
"""
        for i, ev in enumerate(evidence, 1):
            md_report += f"\n### Match {i} (Cosine Similarity: {ev['similarity']:.3f})\n"
            md_report += f"- **Job Excerpt:** {ev['job_requirement_excerpt']}\n"
            md_report += f"- **Resume Excerpt:** {ev['resume_evidence_excerpt']}\n"

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"resume_match_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                label="📄 Download Markdown Summary",
                data=md_report,
                file_name=f"resume_match_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.expander("🔍 View Raw JSON Payload"):
            st.json(report_dict)

else:
    # Initial empty state guidance
    st.markdown(
        """
        <div style="text-align:center; padding:36px 20px; background:#f8fafc; border-radius:12px; border:1px dashed #cbd5e1; margin-top:20px;">
            <div style="font-size:2.5rem; margin-bottom:10px;">📋</div>
            <div style="font-size:1.15rem; font-weight:700; color:#1e293b; margin-bottom:6px;">Ready for Analysis</div>
            <p style="font-size:0.9rem; color:#64748b; max-width:550px; margin:0 auto 16px auto;">
                Upload your resume and job description above, or select <b>⚡ Quick Sample Demo</b> for an instant test drive.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
