from philologic.runtime import concordance_results

from wsgi_helpers import json_endpoint


@json_endpoint
def get_word_frequency(request, config):
    word_frequency_object = generate_word_frequency(request, config)
    return word_frequency_object
