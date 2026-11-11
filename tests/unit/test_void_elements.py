"""Unit tests for empty elements: a <br/> of the text is one line break in the browser, not the two of <br></br>, whose
</br> browsers read as a second <br>; and those written with no slash, as <pb n="2">, hold none of the text after"""

import pytest

from philologic.runtime.ObjectFormatter import format_concordance

from .test_page_marker_display import TOKEN_REGEX, empty_caches, make_database, navigate  # noqa: F401

VERSE = "<p>\nFirst line of verse<br/>\nSecond line of verse<br/>\nThird line\n</p>\n"


@pytest.mark.unit
def test_the_text_page(tmp_path):
    html = navigate(make_database(tmp_path, VERSE, page_marker_regex=None, parse=False), philo_id="1 1")["text"]
    assert html.count("<br/>") == 2
    assert "</br>" not in html


@pytest.mark.unit
def test_concordances():
    formatter = format_concordance
    text = VERSE.encode()
    html = formatter(text, TOKEN_REGEX, [text.index(b"Second")])
    assert html.count("<br/>") == 2
    assert "</br>" not in html
    assert '<span class="highlight">Second</span>' in html


UNCLOSED = '<p>Before the break <pb n="2">after the break<br>next line <milestone type="runhead" n="Head">and on.</p>\n'


@pytest.mark.unit
def test_unclosed_on_the_text_page(tmp_path):
    html = navigate(make_database(tmp_path, UNCLOSED, page_marker_regex=None, parse=False), philo_id="1 1")["text"]
    assert '<span n="2" class="xml-pb-image">2</span>after the break<br/>next line ' in html
    assert '<span type="runhead" n="Head" class="xml-milestone"></span>and on.' in html


@pytest.mark.unit
def test_unclosed_in_concordances():
    formatter = format_concordance
    text = UNCLOSED.encode()
    html = formatter(text, TOKEN_REGEX, [text.index(b"next")])
    assert '<br/><span class="highlight">next</span> line' in html
    assert 'n="Head"></span>and on.' in html
