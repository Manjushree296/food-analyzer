import easyocr  # type: ignore
from deep_translator import GoogleTranslator  # type: ignore

_reader = None
def get_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['ja', 'en'], gpu=False)
    return _reader

def extract_text(image_path: str) -> str:
    reader = get_reader()
    result = reader.readtext(image_path, detail=0)
    text = " ".join(result)
    if text.strip() == "":
        return ""
    # Translate Japanese to English
    try:
        text_en = GoogleTranslator(source='ja', target='en').translate(text)
        return text_en
    except:
        return text
