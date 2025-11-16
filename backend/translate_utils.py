from deep_translator import GoogleTranslator

def translate_list(text_list, source='ja', target='en'):
    return [GoogleTranslator(source=source, target=target).translate(t) for t in text_list]
