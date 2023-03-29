"""
Original implementation directly parsing crawled data.
"""

import logging
import math
import os
import pickle
import xml.etree.ElementTree as ET
from itertools import chain, groupby, repeat

from mfp.data.crello import schema

NS = {
    "svg": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
    "xhtml": "http://www.w3.org/1999/xhtml",
}
ET.register_namespace("", NS["svg"])
ET.register_namespace("xlink", NS["xlink"])
ET.register_namespace("html", NS["xhtml"])

logger = logging.getLogger(__name__)

# DUMMY_TEXT = '''
# Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
# incididunt ut labore et dolore magna aliqua.
# '''
DUMMY_TEXT = """
TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT TEXT
"""

PKL_DIR = f"{os.path.dirname(__file__)}/../../../../data/crello/pkls"


def load_fonts_css(path: str):
    """
    Load font-family to stylesheet rules mapping.
    Get css from
    """
    import tinycss

    parser = tinycss.make_parser("fonts3")
    stylesheet = parser.parse_stylesheet_file(path)
    faces = [
        {
            decl.name: decl.value.as_css().replace("_old", "")
            for decl in rule.declarations
        }
        for rule in stylesheet.rules
    ]
    return {
        face: list(it) for face, it in groupby(faces, lambda x: x.get("font-family"))
    }


class SVGBuilder(object):
    """
    Utility to generate SVG for visualization.

    Usage::

        dataspec = DataSpec(...)
        dataset = dataspec.make_dataset('val')
        example = next(iter(dataset))

        # Manual colormap.
        builder = SVGBuilder(
            'type',
            colormap={
                '': 'none',
                'svgElement': 'blue',
                'textElement': 'red',
                'imageElement': 'green',
                'maskElement': 'cyan',
                'coloredBackground': 'magenta',
                'videoElement': 'yellow',
            },
            max_width=144,
        )
        for item in dataspec.unbatch(example):
            svg = builder(item)

        # Auto colormap by preprocessor.
        builder = SVGBuilder(
            'component',
            preprocessor=dataspec.preprocessor,
            max_width=144,
        )
        for item in dataspec.unbatch(example):
            svg = builder(item)

    """

    def __init__(
        self,
        key=None,
        preprocessor=None,
        colormap=None,
        canvas_width=None,
        canvas_height=None,
        max_width=None,
        max_height=None,
        opacity=0.5,
        image_db=None,
        text_db=None,
        render_text=False,
        **kwargs,
    ):
        assert key
        self._key = key
        self._canvas_width = canvas_width or 256
        self._canvas_height = canvas_height or 256
        self._max_width = max_width
        self._max_height = max_height
        self._opacity = opacity
        self._render_text = render_text
        assert preprocessor or colormap
        if preprocessor is None or key == "color":
            self._colormap = colormap
        else:
            vocabulary = preprocessor[key].get_vocabulary()
            self._colormap = self._make_colormap(vocabulary, colormap)
        self._image_db = image_db
        self._text_db = text_db
        self.fonts = load_fonts_css(
            os.path.dirname(__file__) + "/../data/crello/fonts.css"
        )

    def __call__(self, document):
        canvas_width, canvas_height = self.compute_canvas_size(document)
        root = ET.Element(
            ET.QName(NS["svg"], "svg"),
            {
                "width": str(canvas_width),
                "height": str(canvas_height),
                "viewBox": "0 0 1 1",
                # 'style': 'background-color: #EEE',
                "style": "background-color: #FFF",
                "preserveAspectRatio": "none",
            },
        )

        doc_size = {
            "width": document["canvas_width"],
            "height": document["canvas_height"],
        }

        # load pickled data
        id_ = document["id"].decode()
        pkl_file = f"{PKL_DIR}/{id_[:3]}/{id_}.pkl"
        with open(pkl_file, "rb") as f:
            pkl_data = pickle.load(f)
        pkl_elements = pkl_data.template[0].elements
        pkl_uuids = [e.uuid for e in pkl_elements]

        if len(pkl_elements) != len(document["elements"]):
            plen = len(pkl_elements)
            elen = len(document["elements"])
            logger.warning(f"#elements mismatch {plen},{elen} for pkl, tfr")

        # find one-to-one correspondense
        doc2pkl = {}
        for i, element in enumerate(document["elements"]):
            uuid_ = element["uuid"].decode()
            try:
                doc2pkl[i] = pkl_uuids.index(uuid_)
            except ValueError:
                logger.warning(f"Not found: {uuid_}")

        for i, element in enumerate(document["elements"]):
            if i not in doc2pkl:  # with very low prob. it cannot be found
                continue
            if self._key == "color":
                fill = "rgb(%g,%g,%g)" % tuple(map(int, element["color"]))
            else:
                fill = self._colormap.get(element[self._key], "none")

            image_url = ""
            if self._image_db:
                if (
                    element.get(self._image_db.condition["key"])
                    in self._image_db.condition["values"]
                ):
                    image_url = self._image_db.search(element[self._image_db.value])

            if self._text_db:
                if (
                    element.get(self._text_db.condition["key"])
                    in self._text_db.condition["values"]
                ):
                    text = self._text_db.search(element[self._text_db.value])
                else:
                    text = DUMMY_TEXT
            else:
                text = DUMMY_TEXT

            if image_url:
                node = self._make_image(root, element, image_url)
            elif self._render_text and element.get("type") == "textElement":
                node = self._make_text_element(
                    root,
                    element,
                    fill,
                    doc_size,
                    text,
                    pkl_elements[doc2pkl[i]],
                )
            else:
                node = self._make_rect(root, element, fill)

            title = ET.SubElement(node, ET.QName(NS["svg"], "title"))
            title.text = str(
                {
                    k: v
                    for k, v in element.items()
                    # if not (self._image_db and k == self._image_db.value)
                    # to filter out large array like image/text_embedding
                    if not isinstance(v, list)
                }
            )

        # get links for fonts
        style = ET.SubElement(root, "{%s}style" % NS["svg"])
        self._fill_stylesheet(root, style)

        return ET.tostring(root).decode("utf-8")

    def _fill_stylesheet(self, root, style):
        font_families = {
            text.get("font-family")
            for text in root.iter("{%s}text" % NS["svg"])
            if text.get("font-family") is not None
        }
        style.text = "\n".join(
            "@font-face { %s }" % " ".join("%s: %s;" % (key, item[key]) for key in item)
            for item in chain.from_iterable(
                self.fonts.get(family, []) for family in font_families
            )
        )

    def compute_canvas_size(self, document):
        canvas_width = document.get("canvas_width", self._canvas_width)
        canvas_height = document.get("canvas_height", self._canvas_height)
        scale = 1.0
        if self._max_width is not None:
            scale = min(self._max_width / canvas_width, scale)
        if self._max_height is not None:
            scale = min(self._max_height / canvas_height, scale)
        return canvas_width * scale, canvas_height * scale

    def _make_colormap(self, vocabulary, colormap=None):
        """
        Generate a colormap for the specified vocabulary list.
        """
        from matplotlib import cm

        vocab_size = len(vocabulary)
        cmap = cm.get_cmap(colormap or "tab20", vocab_size)
        return {
            label: "rgb(%g,%g,%g)" % tuple(int(x * 255) for x in c[:3])
            for label, c in zip(vocabulary, cmap(range(vocab_size)))
        }

    def _make_text_element(
        self, parent, element, fill, doc_size, text_str, pkl_element
    ):
        def _make_map(m, default_key=None):
            return chain.from_iterable(
                repeat(
                    (x.get("type", default_key), x["value"]),
                    x["endIndex"] - x["startIndex"],
                )
                for x in m
            )

        def _generate_spans(text, style_map):
            offset = 0
            for style, it in groupby(style_map):
                length = len(list(it)) + 1
                item = dict(style)
                item["text"] = text[offset : offset + length]
                yield item
                offset += length

        def _make_linespans(text, pkl_element):
            style_map = list(
                zip(
                    _make_map(pkl_element.colorMap, default_key="color"),
                    _make_map(pkl_element.boldMap, default_key="bold"),
                    _make_map(pkl_element.italicMap, default_key="italic"),
                )
            )
            br_inds = [i for (i, t) in enumerate(text) if t == "\n"]

            if pkl_element.lineMap is not None:
                default_line_map = []
            elif len(br_inds) == 0:
                default_line_map = [
                    {"startIndex": 0, "endIndex": len(pkl_element.text)}
                ]
            else:
                default_line_map = []
                start = 0
                for ind in br_inds:
                    default_line_map.append({"startIndex": start, "endIndex": ind - 1})
                    start = ind + 1
                default_line_map.append(
                    {"startIndex": start, "endIndex": len(text) - 1}
                )

            for line in pkl_element.lineMap or default_line_map:
                start = line["startIndex"]
                end = line["endIndex"] + 1

                line_text = text[start:end]
                line_style_map = style_map[start:end]
                yield _generate_spans(line_text, line_style_map)

        margin = element["height"] * 0.1  # To avoid unexpected clipping.
        container = ET.SubElement(
            parent,
            ET.QName(NS["svg"], "svg"),
            {
                "id": element["uuid"].decode(),
                "class": element["type"],
                "x": "%g" % (element["left"] or 0),
                "y": "%g" % ((element["top"] or 0) - margin),
                "width": "%g" % (element["width"]),
                "overflow": "visible",
            },
        )
        opacity = element.get("opacity", 1.0)
        if opacity < 1:
            container.set("opacity", "%g" % opacity)

        # in element filling, different type might be used
        # in that case, we should somehow feed default values
        font_size = (
            getattr(pkl_element, "fontSize", doc_size["height"]) / doc_size["height"]
        )
        text_align = getattr(pkl_element, "textAlign", "center")
        line_height = getattr(pkl_element, "lineHeight", 1.0)
        capitalize = getattr(pkl_element, "capitalize", False)
        underline = getattr(pkl_element, "underline", False)
        letter_spacing = getattr(pkl_element, "letterSpacing", doc_size["width"])
        if letter_spacing is None:
            letter_spacing = 0.0
        else:
            letter_spacing /= doc_size["width"]

        if not getattr(pkl_element, "lineMap", False):
            setattr(pkl_element, "lineMap", None)
        if not getattr(pkl_element, "colorMap", False):
            setattr(pkl_element, "colorMap", [])
        if not getattr(pkl_element, "boldMap", False):
            setattr(pkl_element, "boldMap", [])
        if not getattr(pkl_element, "italicMap", False):
            setattr(pkl_element, "italicMap", [])
        if not getattr(pkl_element, "text", False):
            setattr(pkl_element, "text", "a" * 1000)

        text = ET.SubElement(
            container,
            "{%s}text" % NS["svg"],
            {
                "font-size": "%g" % font_size,
                "font-family": element["font_family"],
                "letter-spacing": "%g" % letter_spacing,
            },
        )

        if underline:
            text.set("text-decoration", "underline")
        if pkl_element.angle is not None and pkl_element.angle != 0:
            # Note: Chromium clips the svg region.
            angle = 180 * (pkl_element.angle / math.pi)
            text.set(
                "transform",
                "rotate(%g, %g, %g)"
                % (angle, element["width"] / 2, element["height"] / 2),
            )
        x = {"left": "0", "center": "50%", "right": "100%"}[text_align]
        anchor = {"left": "start", "center": "middle", "right": "end"}[text_align]

        line_height = line_height * font_size

        # print('L343', fill)
        for index, line in enumerate(_make_linespans(text_str, pkl_element)):
            line_tspan = ET.SubElement(
                text,
                "{%s}tspan" % NS["svg"],
                {
                    "dy": "%g" % line_height,
                    "x": x,
                    "text-anchor": anchor,
                    "dominant-baseline": "central",
                },
            )
            if index == 0:
                text.set("y", "%g" % (margin))
                line_tspan.set("dy", "%g" % (line_height / 2))
            for span in line:

                def f(x):
                    # convert 'rgb(255,255,255)' to 'FFFFFF'
                    values = [
                        "{:02x}".format(int(s)).upper() for s in x[4:-1].split(",")
                    ]
                    return "".join(values)

                color = f(fill)

                # print('L359', span['color'])
                tspan = ET.SubElement(
                    line_tspan,
                    "{%s}tspan" % NS["svg"],
                    {
                        # 'fill': '#%s' % span['color'],
                        "fill": "#%s" % color,
                        "dominant-baseline": "central",
                    },
                )
                tspan.text = span["text"].strip()
                if span["bold"]:
                    tspan.set("font-weight", "bold")
                if span["italic"]:
                    tspan.set("font-style", "italic")
                if capitalize:
                    # Capitalize at the leaf span for Safari compatibility.
                    tspan.set("style", "text-transform: uppercase;")

        return container

    def _make_image(self, parent, element, image_url):
        return ET.SubElement(
            parent,
            ET.QName(NS["svg"], "image"),
            {
                "x": str(element["left"]),
                "y": str(element["top"]),
                "width": str(element["width"]),
                "height": str(element["height"]),
                ET.QName(NS["xlink"], "href"): image_url,
                "opacity": str(element.get("opacity", 1.0)),
                "preserveAspectRatio": "none",
            },
        )

    def _make_rect(self, parent, element, fill):
        return ET.SubElement(
            parent,
            ET.QName(NS["svg"], "rect"),
            {
                "x": str(element["left"]),
                "y": str(element["top"]),
                "width": str(element["width"]),
                "height": str(element["height"]),
                "fill": str(fill),
                "opacity": str(element.get("opacity", 1.0) * self._opacity),
            },
        )
