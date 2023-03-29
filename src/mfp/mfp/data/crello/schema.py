"""
Inferred crello template schema from JSON. Could be wrong.


Templates having elements with `originalId` field are old and should be ignored.

It seems `innerId` specifies a big schema change. Name this `V1` and `V2`.



The following is the list of leaf element types.

    SVGElementV1
    SVGElementV2
    ImageElementV1
    ImageElementV2
    MaskElementV1
    MaskElementV2
    TextMaskElementV2
    TextElementV1
    TextElementV2
    ColoredBackgroundV1
    ColoredBackgroundV2
    PersistGroupElement
    GroupElement

"""
import dataclasses
import json
import logging
from typing import Any, Dict, List, Optional, Union

import dacite

logger = logging.getLogger(__name__)


class _FromDictMixin(object):
    @classmethod
    def from_dict(kls, value):
        if "originalId" in value:
            raise NotImplementedError("Old format not supported.")
        type_ = value.get("type")
        if type_ == "videoElement":
            raise NotImplementedError("Video not supported.")
        has_inner_id = "innerId" in value
        true_class = ElementTypes.get((type_, has_inner_id), kls)
        if true_class == MaskElementV2 and value.get("maskType"):
            true_class = TextMaskElementV2
        return dacite.from_dict(
            true_class,
            value,
            dacite.Config(
                type_hooks={
                    BaseElement: BaseElement.from_dict,
                }
            ),
        )

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class BaseElement(_FromDictMixin):
    angle: float
    height: float
    opacity: float
    position: Dict[str, float]
    type: str
    uuid: str
    width: float

    dphId: Optional[str]
    isBackground: Optional[bool]
    isFreeItem: Optional[bool]
    shouldIgnoreMountingStash: Optional[bool]
    templateAsset: Optional[bool]


@dataclasses.dataclass
class _V1(object):
    """V1 schema"""

    forSubscribers: Optional[bool]


@dataclasses.dataclass
class _V2(object):
    """V2 schema"""

    innerId: int
    originalAngle: float

    mediaAsset: Optional[bool]
    isSharedProject: Optional[bool]
    hasBackgroundLabel: Optional[bool]
    renderMode: Optional[bool]
    highlighted: Optional[bool]
    selected: Optional[bool]
    multiSelectActive: Optional[bool]


@dataclasses.dataclass
class SVGElement(BaseElement):
    colors: List[Dict[str, Any]]
    mediaId: str


@dataclasses.dataclass
class SVGElementV1(SVGElement, _V1):
    flipHorizontal: int
    flipVertical: int
    left: float
    locked: bool
    top: float


@dataclasses.dataclass
class SVGElementV2(SVGElement, _V2):
    flipped: Dict[str, int]
    id: Union[str, int]
    isBackground2: bool

    special: Optional[bool]
    newColors: Optional[List[Any]]
    url: Optional[str]
    locked: Optional[bool]


@dataclasses.dataclass
class ImageElement(BaseElement):
    cropOpts: Dict[str, Any]
    filters: Dict[str, float]
    filtersIntensityEnabled: bool
    filtersPresetIntensity: int
    mediaId: str
    originalImageWidth: float
    originalImageHeight: float


@dataclasses.dataclass
class ImageElementV1(ImageElement, _V1):
    flipHorizontal: int
    flipVertical: int
    left: float
    top: float
    locked: Optional[bool]  # A very few samples
    filtersPreset: Optional[Union[bool, str]]


@dataclasses.dataclass
class ImageElementV2(ImageElement, _V2):
    filtersPreset: Union[bool, str]
    flipped: Dict[str, int]
    id: Union[str, int]
    isBackground2: bool

    cropOverlay: Optional[Any]
    cropping: Optional[bool]
    sticky: Optional[bool]
    originalSrc: Optional[str]
    src: Optional[str]
    dummyHolder: Optional[Any]
    locked: Optional[bool]
    finalize: Optional[bool]

    # The following are rare.
    _resizing: Optional[bool]
    draggable: Optional[bool]
    el: Optional[Dict[str, Any]]
    filtersStyle: Optional[Any]
    image: Optional[Any]
    initialCropOpts: Optional[Dict[str, Any]]
    overlay: Optional[Any]
    resizablePoints: Optional[List[Dict[str, Any]]]
    style: Optional[Dict[str, Any]]
    underlyingEl: Optional[Dict[str, Any]]
    url: Optional[str]


@dataclasses.dataclass
class TextElement(BaseElement):
    boldMap: List[Dict[str, Any]]
    capitalize: bool
    colorMap: List[Dict[str, Any]]
    font: str
    fontSize: float
    italicMap: List[Dict[str, Any]]
    letterSpacing: float
    lineHeight: float
    text: str
    textAlign: str
    lineMap: Optional[List[Dict[str, Any]]]


@dataclasses.dataclass
class TextElementV1(TextElement, _V1):
    left: float
    locked: bool
    top: float
    underline: bool
    wordBreak: str


@dataclasses.dataclass
class TextElementV2(TextElement, _V2):
    id: Optional[Union[str, int]]
    underline: Optional[bool]
    wordBreak: Optional[str]
    patched: Optional[bool]
    __disableUnlockSelection: Optional[bool]
    isBackground2: Optional[bool]
    drag_lock: Optional[bool]
    __selectionInProgress: Optional[bool]
    locked: Optional[bool]


@dataclasses.dataclass
class MaskElement(BaseElement):
    filters: Dict[str, float]
    filtersIntensityEnabled: bool
    filtersPresetIntensity: int


@dataclasses.dataclass
class MaskElementV1(MaskElement, _V1):
    colors: List[Dict[str, Any]]
    elements: List[BaseElement]
    flipHorizontal: int
    flipVertical: int
    left: float
    locked: bool
    maskData: Dict[str, Any]
    mediaId: str
    maskType: str
    top: float
    filtersPreset: Optional[Union[bool, str]]

    def __iter__(self):
        yield from self.elements


@dataclasses.dataclass
class MaskElementV2(MaskElement, _V2):
    filtersPreset: Union[bool, str]
    filtersStyle: Dict[str, Any]
    flipped: Dict[str, int]
    id: Union[str, int]
    imagePosition: Dict[str, float]
    presetPic: bool
    stickedImageData: Dict[str, Any]
    mediaId: str

    colors: Optional[List[Dict[str, Any]]]
    cropping: Optional[bool]
    finalize: Optional[bool]
    url: Optional[str]


@dataclasses.dataclass
class TextMaskElementV2(MaskElement, _V2):
    filtersPreset: Union[bool, str]
    filtersStyle: Dict[str, Any]
    flipped: Dict[str, int]
    imagePosition: Dict[str, float]
    maskData: Dict[str, Any]
    maskType: str  # Always 'text'
    presetPic: bool
    stickedImageData: Dict[str, Any]
    cropping: Optional[bool]
    locked: Optional[bool]


@dataclasses.dataclass
class ColoredBackgroundV1(BaseElement, _V1):
    colors: List[Dict[str, Any]]
    left: float
    locked: bool
    top: float


@dataclasses.dataclass
class ColoredBackgroundV2(BaseElement, _V2):
    id: int
    colors: Optional[List[Dict[str, Any]]]
    url: Optional[str]
    backgroundColor: Optional[str]  # A few has this instead of colors.


@dataclasses.dataclass
class PersistGroupElement(BaseElement):
    elements: List[BaseElement]
    left: float
    locked: bool
    top: float

    def __iter__(self):
        yield from self.elements


@dataclasses.dataclass
class GroupElement(BaseElement, _V2):
    children: List[BaseElement]
    el: Optional[Dict[str, Any]]
    draggable: Optional[bool]
    locked: Optional[bool]
    overlay: Optional[Any]
    resizablePoints: Optional[List[Dict[str, Any]]]
    style: Optional[Dict[str, Any]]
    underlyingEl: Optional[Dict[str, Any]]

    @property
    def elements(self):
        return self.children

    def __iter__(self):
        yield from self.children


@dataclasses.dataclass
class Page(_FromDictMixin):
    animationMode: Optional[str]
    customAnimationDuration: Optional[int]
    elements: List[BaseElement]
    uuid: Optional[str]
    audio: Optional[List[Any]]
    pageNumber: Optional[int]

    def __iter__(self):
        yield from self.elements


@dataclasses.dataclass
class Template(_FromDictMixin):
    id: str
    name: Optional[str]
    width: Union[float, str]
    height: Union[float, str]
    category: Optional[Union[str, List[str]]]
    categoryCaption: str
    group: str
    measureUnits: str
    driveFileIds: Optional[List[str]]
    template: List[Page]
    templateType: str
    hash: str
    pixelWidth: float
    pixelHeight: float
    previewImageUrls: List[str]
    hasAnimatedPreview: bool
    hasAnimatedScreenPreview: bool
    downloadUrl: Optional[str]
    status: str
    format: str
    suitability: List[str]
    folder: Optional[str]
    errors: Optional[List[Any]]
    warnings: Optional[List[Any]]
    studioName: str
    createdAt: int
    updatedAt: int
    acceptedAt: int
    attributedAt: Optional[int]
    keywords: Optional[Dict[str, Any]]
    industries: Optional[List[str]]
    languages: Optional[List[str]]
    title: Optional[str]
    localizedTitle: Optional[Dict[str, str]]
    forSubscribers: Optional[bool]
    v2: Optional[bool]

    @property
    def url(self):
        return "https://crello.com/artboard/?template=" + self.id

    def __iter__(self):
        yield from self.template

    def iter_elements(self):
        for element in self.template[0]:
            yield element
            if hasattr(element, "__iter__"):
                yield from element

    @staticmethod
    def load(path, raw_format=False):
        import glob
        import json

        def _iter(path):
            skipped = 0
            for file_name in glob.glob(path):
                with open(file_name, "r") as f:
                    for line in f:
                        value = json.loads(line)
                        if raw_format:
                            yield value
                        else:
                            if value.get("templateType") != "regular":
                                continue
                            try:
                                yield Template.from_dict(value)
                            except NotImplementedError as e:
                                skipped += 1
            if skipped:
                logger.info("Skipped %d templates" % skipped)

        return list(_iter(path))


ElementTypes = {
    ("svgElement", False): SVGElementV1,
    ("svgElement", True): SVGElementV2,
    ("imageElement", False): ImageElementV1,
    ("imageElement", True): ImageElementV2,
    ("maskElement", False): MaskElementV1,
    ("maskElement", True): MaskElementV2,
    ("textElement", False): TextElementV1,
    ("textElement", True): TextElementV2,
    ("coloredBackground", False): ColoredBackgroundV1,
    ("coloredBackground", True): ColoredBackgroundV2,
    ("persistGroupElement", False): PersistGroupElement,
    ("groupElement", True): GroupElement,
}
