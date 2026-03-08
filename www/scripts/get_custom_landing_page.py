import os


def get_custom_landing_page(request, config):
    """Read and return the custom landing page HTML file from the app directory."""
    custom_page = config.landing_page_browsing
    file_path = os.path.realpath(os.path.join(config.db_path, custom_page))
    # Path traversal protection
    if not file_path.startswith(os.path.realpath(config.db_path)):
        return ""
    if not os.path.isfile(file_path):
        return ""
    with open(file_path) as f:
        return f.read()
