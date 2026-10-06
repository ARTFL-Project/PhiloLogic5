#!/var/lib/philologic5/philologic_env/bin/python3


from html.entities import name2codepoint

import regex as re

entities_match = re.compile(r"&#?\w+;")

# Entities XML defines itself
XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}


def convert_entities(text, keep_xml_entities=False):
    """Convert entities. With keep_xml_entities, leave those an XML parser decodes itself (character references
    and XML_ENTITIES): decoding &amp; or &lt; before parsing leaves a bare & or <, which breaks the XML."""

    def fixup(m):
        text = m.group(0)
        if keep_xml_entities and (text[:2] == "&#" or text[1:-1] in XML_ENTITIES):
            return text
        if text[:2] == "&#":
            # character reference
            try:
                if text[:3] == "&#x":
                    return chr(int(text[3:-1], 16))
                else:
                    return chr(int(text[2:-1]))
            except ValueError:
                pass
        else:
            # named entity
            try:
                text = chr(name2codepoint[text[1:-1]])
            except KeyError:
                pass
        return text  # leave as is

    return entities_match.sub(fixup, text)
