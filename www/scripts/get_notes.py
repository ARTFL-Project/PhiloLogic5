from philologic.runtime import generate_text_object

def get_notes(request, config):
    return generate_text_object(request, config, note=True)
