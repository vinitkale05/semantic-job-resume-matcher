# Semantic Job–Resume Matcher Using Generative AI

A Python and Streamlit project that compares resume text with a job description using sentence embeddings, cosine similarity, skill extraction, and optional Gemini-generated feedback.

## Features

- PDF and DOCX resume text extraction; paste-text input is also supported.
- Sentence Transformer embeddings using `all-MiniLM-L6-v2`.
- Chunk-level semantic similarity and closest-text evidence.
- Skill extraction and explicit skill-gap comparison.
- Optional Gemini recommendations and resume-bullet improvement.
- Interactive dashboard and downloadable JSON report.
- Unit tests and sample input files.

## How it works

1. Text is extracted from a resume and job description.
2. Text is divided into chunks and encoded with a Sentence Transformer.
3. Each job-description chunk is compared to resume chunks using cosine similarity. The app averages each job chunk's best match and displays that result on a 0–100 scale.
4. A skill dictionary detects explicit mentions of skills and identifies overlap and gaps.
5. If Gemini is configured, an LLM generates grounded explanations and recommendations.

The similarity score is **not** a probability of being hired. Skill extraction is dictionary-based and may miss synonyms or context. Missing-from-resume skills are not proof that a candidate lacks them. Always review results manually.

## Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- Internet access for installing dependencies and downloading the embedding model on first run
- Optional Gemini API key for generated feedback

## macOS installation

From Terminal, change to the project directory:

```bash
cd "/path/to/semantic-job-resume-matcher"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Optional Gemini configuration

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
GEMINI_API_KEY=your_real_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Choose a model supported by your Gemini API account. Never commit `.env` or expose the API key. Semantic matching works without the key; generated feedback does not.

## Windows installation

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Run tests

```bash
python -m pytest -q
```

The semantic algorithm tests use small vectors and do not download the embedding model. Parser tests create an in-memory DOCX file.

## Project structure

```text
app.py
requirements.txt
README.md
.env.example
.gitignore
modules/
  resume_parser.py
  text_preprocessor.py
  semantic_matcher.py
  skill_extractor.py
  genai_analyzer.py
  resume_improver.py
  report_generator.py
  validators.py
utils/
tests/
sample_data/
```

## Privacy and responsible use

The app does not save uploaded resumes by default. If Gemini analysis is enabled, relevant resume and job-description text is sent to the configured API provider. Do not use confidential resumes unless permitted by the data subject and your institution's policies. This tool should support human review, not make automated hiring decisions.

## Future enhancements

- Add richer skill taxonomies and synonym handling.
- Evaluate against manually labeled resume/job pairs with precision, recall, F1, and ranking metrics.
- Add configurable weights for required skills, experience, and semantic evidence.
- Support multiple job descriptions and exportable PDF reports.
- Add multilingual resume support and optional local-only LLM inference.
