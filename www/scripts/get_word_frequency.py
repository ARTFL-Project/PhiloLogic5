from philologic.runtime import concordance_results

def get_word_frequency(request, config):
    word_frequency_object = generate_word_frequency(request, config)
    return word_frequency_object
