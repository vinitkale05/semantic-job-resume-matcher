from io import BytesIO
from modules.validators import validate_text, validate_upload

class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data
    def getvalue(self):
        return self._data

def test_validate_text():
    assert validate_text("This is a sufficiently long sample text.")[0]
    assert not validate_text("short")[0]

def test_validate_upload():
    assert validate_upload(FakeUpload("resume.pdf", b"pdf bytes"))[0]
    assert not validate_upload(FakeUpload("resume.exe", b"x"))[0]
    assert not validate_upload(FakeUpload("resume.pdf", b""))[0]
