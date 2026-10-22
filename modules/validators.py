"""Input and upload validation helpers."""

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_RESUME_TYPES = {"pdf", "docx"}


def validate_upload(uploaded_file, max_bytes: int = MAX_UPLOAD_BYTES) -> tuple[bool, str]:
    """Validate a Streamlit UploadedFile by extension and size."""
    if uploaded_file is None:
        return False, "No file was provided."
    name = getattr(uploaded_file, "name", "")
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in ALLOWED_RESUME_TYPES:
        return False, "Unsupported file type. Upload a PDF or DOCX file."
    data = uploaded_file.getvalue()
    if not data:
        return False, "The uploaded file is empty."
    if len(data) > max_bytes:
        return False, "The file exceeds the 5 MB size limit."
    return True, "File is valid."


def validate_text(text: str, label: str = "Text", min_chars: int = 30) -> tuple[bool, str]:
    if not isinstance(text, str) or len(text.strip()) < min_chars:
        return False, f"{label} must contain at least {min_chars} characters."
    return True, "Text is valid."
