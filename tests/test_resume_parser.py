from io import BytesIO
import pytest
from docx import Document
from modules.resume_parser import parse_resume

class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data
    def getvalue(self):
        return self._data

def test_parse_docx():
    buffer = BytesIO()
    doc = Document()
    doc.add_paragraph("Python and machine learning experience")
    doc.save(buffer)
    result = parse_resume(FakeUpload("resume.docx", buffer.getvalue()))
    assert "Python" in result

def test_unsupported_format():
    with pytest.raises(ValueError):
        parse_resume(FakeUpload("resume.txt", b"plain text"))
