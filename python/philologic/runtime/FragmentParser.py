#!/var/lib/philologic5/philologic_env/bin/python3


from lxml import etree
from philologic import shlaxtree as st


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


def strip_tags(text):
    parser = FragmentStripper()
    feeder = st.ShlaxIngestor(target=parser)
    feeder.feed(text)
    return feeder.close()
