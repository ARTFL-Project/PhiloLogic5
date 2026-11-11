#!/var/lib/philologic5/philologic_env/bin/python3


from lxml import etree
from philologic import shlaxtree as st

# Elements always empty, even when a text writes <pb n="2"> with no slash: what follows is their tail, not their content
EMPTY_TAGS = frozenset(("pb", "lb", "cb", "milestone", "br"))


class FragmentParser:
    """Fragment Parser: reconstructs broken trees"""

    def __init__(self):
        self.root = etree.Element("div", {"class": "philologic-fragment"})
        self.root.text = ""
        self.root.tail = ""
        self.current_el = self.root
        self.current_tail = None
        self.in_tag = True
        self.stack = []

    def feed(self, kind, content, offset, name, attributes):
        """Take the events of the ShlaxIngestor"""
        if kind == "start":
            # Double-quoted empty values once came up as None: lxml takes an empty string
            self.start(name, {k: "" if v is None else v for k, v in attributes.items()})
        elif kind == "end":
            self.end(name)
        elif kind == "text":
            self.data(content)

    def start(self, tag, attrib):
        empty = (tag[tag.index(":") + 1 :] if ":" in tag else tag) in EMPTY_TAGS
        if not empty:
            self.stack.append(tag)
        # Without their namespace prefix, up to the first ":", with the values they had before any was renamed
        for k, v in [(k, v) for k, v in attrib.items() if ":" in k]:
            del attrib[k]
            attrib[k[k.index(":") + 1 :]] = v
        # Without its namespace prefix, as attribute names: lxml takes no "jx:cl" for a tag name
        if ":" in tag:
            tag = tag[tag.index(":") + 1 :]
        new_el = etree.SubElement(self.current_el, tag, attrib)
        new_el.text = ""
        new_el.tail = ""
        if empty:
            self.current_tail = new_el
            return
        self.current_el = new_el
        self.in_tag = True
        self.current_tail = None

    def end(self, tag):
        if len(self.stack) and self.stack[-1] == tag:
            self.current_tail = self.current_el
            self.stack.pop()
            self.current_el = self.current_el.getparent()
            self.in_tag = False

        else:
            pass

    def data(self, data):
        if self.current_tail is not None:
            self.current_tail.tail += data
        else:
            self.current_el.text += data

    def comment(self, text):
        pass

    def close(self):
        self.stack.reverse()
        for s in self.stack:
            self.end(s)
        r = self.root
        self.stack = []
        return r


class LenientFragmentParser(FragmentParser):
    """A FragmentParser keeping as text the tags lxml makes no element of, such as the "<abducere: " of plain-text
    dictionaries, whose name is nothing once its "prefix" is dropped. They are kept escaped, as the HTML fallback has
    the "<" it takes for no tag: the formatters decode the entities of the text once more."""

    def feed(self, kind, content, offset, name, attributes):
        if kind != "start":
            return super().feed(kind, content, offset, name, attributes)
        depth = len(self.stack)
        try:
            super().feed(kind, content, offset, name, attributes)
        except ValueError:
            del self.stack[depth:]
            self.data(content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class FragmentStripper:
    def __init__(self):
        self.buffer = ""

    def feed(self, *event):
        (kind, content, offset, name, attributes) = event
        if kind == "text":
            self.buffer += content

    def close(self):
        return self.buffer


def parse(text) -> etree.Element:
    try:
        parser = FragmentParser()
        feeder = st.ShlaxIngestor(target=parser)
        feeder.feed(text)
        return feeder.close()
    except ValueError:
        pass
    try:
        # we use LXML's HTML parser which is more flexible and then feed the result to fragment parser
        parser = etree.HTMLParser()
        tree = etree.fromstring(text, parser=parser)
        new_text = etree.tostring(tree, method="xml")
        if isinstance(new_text, bytes):
            new_text = new_text.decode("utf8", "ignore")
        new_text = (
            new_text.replace("<html><body>", "")
            .replace("</body></html>", "")
            .replace("philohighlight", "philoHighlight")
        )
        parser = FragmentParser()
        feeder = st.ShlaxIngestor(target=parser)
        feeder.feed(new_text)
        return feeder.close()
    except ValueError:
        # the HTML parser kept the tag lxml makes no element of: the text with that tag as text
        feeder = st.ShlaxIngestor(target=LenientFragmentParser())
        feeder.feed(text)
        return feeder.close()


def strip_tags(text):
    parser = FragmentStripper()
    feeder = st.ShlaxIngestor(target=parser)
    feeder.feed(text)
    return feeder.close()
