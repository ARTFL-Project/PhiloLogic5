---
title: Text Encoding Specification
---

-   PhiloLogic can parse non-valid XML and does not rely on an XML lib for document parsing (with the exception of the TEI Header).

-   The only requirement is that files are encoded in **UTF-8**.

-   We mostly support the TEI spec, though we only index a small reasonable subset of the spec.

-   We only support double quotes for attributes, such as `<pb n="1"/>`.<br>
    In other words, we do **NOT** support `<pb n='1'/>`.

### Page tags

-   Pages are encoded with the `<pb>` tag.
-   Page image filenames should be stored inside the facs attribute, such as in the example below:

```xml
<pb facs="ENC_23-1-1.jpeg"/>
```

-   You can also specify multiple images separated by a space such as below:

```xml
<pb facs="small/ENC_23-1-1.jpeg large/ENC_23-1-1.jpeg"/>
```

This will produce produce a link to the first image, the second one will be displayed if clicked on the arrow link in the page turner.

**Note**: The values specified in `facs` must be the complete relative link of the image(s). These are then appended to the url defined in web_config.cfg under `pages_images_url_root`

-   Page numbers should be stored in the n attribute, such as below:

```xml
<pb n="23"/>
```

A page tag with both attributes could look like this:

```xml
<pb n="23" facs="V23/ENC_23-1-1.jpeg"/>
```

### Inline Images

-   Inline images should use the `<graphic>` tag.
-   Links to images should be stored in the facs attribute such as below. Image links should be separated by a space:

```xml
<graphic facs="V23/plate_23_2_2.jpeg"/>
<graphic facs="V23/plate_23_2_2-sm.jpeg V23/plate_23_2_2-lg.jpeg"/>
```

**Note**: The values specified in `facs` must be the complete relative link of the image(s). These are then appended to the url defined in web_config.cfg under `pages_images_url_root`

### External Images

External image are images that should not be rendered alongside the text like inline images. Instead, it should be rendered as an HTML anchor tag with accompanying text.

-   External Images should use the `<ptr>`tag.
-   Links to the image should be stored in the facs attribute such as below. Only one link should be available.
-   The text accompanying the image should be stored in the rend attribute.

```xml
<ptr facs="0000c.jpg" rend="[000c]"/>
```

### Notes

IMPORTANT: While PhiloLogic will display inline notes, it really only properly supports notes
that are divided into the pointer to the note inside the running text, and the note
itself at the end of a text object or of the document.

#### Pointers to notes

-   Pointers to notes should use the `<ref>` tag
-   The `<ref>` tag should have an attribute type of type "note", such as `type="note"`
-   Pointers reference the actual note using the target attribute, such as `target="n1"`.
-   Pointers will be displayed in the text using the contents of the n attribute, otherwise default to "note".

Example of a `<ref>` tag pointing to a `<note>` tag:

```xml
<ref type="note" target="n1" n="1"/>
```

#### Note tags

-   Notes should be stored at the end of the parent `<div>` element or a the end of the doc inside a `<div type="notes">`
-   Notes themselves are stored in a `<note>` tag.
-   Notes should have an attribute id with the value corresponding to the value of target in the pointer referencing the note.
-   Notes are stored as paragraph elements, therefore all `<p>` tags (or any other paragraph level tag) contained within will be ignored though still displayed.

Example of notes inside a `<div1 type="notes">`

```xml
<div1 type="notes">
  <note id="n1">Contents of note....</note>
  <note id="n2">Contents of note....</note>
</div1>
```

### Cross references

-   Cross-references should use the `<ref>` tag
-   The `<ref>` tag should have an attribute type of type "cross", such as `type="cross"`
-   The type "cross" of `<ref>` triggers direct navigation to the object defined in the id attribute.

Example of a cross-reference:

```xml
<ref type="cross" target="c2">See chapter 2</ref>
```

which references the following object using its id attribute:

```xml
<div2 type="Chapter" id="c2">
```

### Search references

-   Search references should use the `<ref>` tag
-   The `<ref>` tag should have an attribute type of type "search", such as `type="search"`
-   The type "search" of `<ref>` triggers a metadata search of the value defined in the target attribute
-   The target attribute value contains the metadata field and metadata value to be searched separated by a `:`,<br>
    such as `target="who:Hamlett"`

Example of a search reference

```xml
<ref type="search" target="head:Gouverner">Gouverner</ref>
```


### Date tags

You can use <date> tags inside the body of your document. The date will be attached to the parent div element. We only support two attributes for `when` and `value`. You can use either attribute to express the date. The date should be using the ISO date format YYYY-MM-DD.

Example of <date> tags:

```xml
<date when="1999-12-23"/>
<date value="1999-12-23"/>
```

Note that you can also add additional attributes in the <date> tag such as
```xml
<date when="1795-11-01" revdate="10-brumaire-IV">Du 10 Brumaire.</date>
```
However, you will need to tell the parser to extract that information by customizing your load_config.py file.

### Using ISO dates in the TEI header
You can use ISO dates for the pub_date and create_date tags in the TEI header. But in order to make those dates searchable, you need to specify the 'date' type in load_config.py in the metadata_sql_types variable. For instance:

```python
metadata_sql_types = {"create_date": "date"}
```
