import logging
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, Union

import faiss
import numpy as np
import tensorflow as tf
from mfp.data import DataSpec

logger = logging.getLogger(__name__)


class _Retriever(object):
    """Image retriever for visualization."""

    def __init__(
        self,
        path: Path,
        key: str,
        value: str,
        condition: Dict[str, Any] = None,
        dim: int = 512,
        # image_path=None,
        # **kwargs,
    ):
        self._path = path
        # self._dataspec = DataSpec("crello-images", path, **kwargs)
        self._dataspec = None
        self._key = key
        self._value = value
        self._condition = condition
        self._dim = dim

        #  or {
        #     "key": "type",
        #     "values": ("imageElement", "maskElement", "svgElement"),
        # }
        # self._image_path = image_path or os.path.join(self._path, "images")

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    @property
    def condition(self):
        return self._condition

    def build(self, split="train"):
        """Build index."""
        logger.info("Fetching image embeddings...")
        dataset = self._dataspec.make_dataset(split)

        # Deduplicate entries.
        d = {}
        for batch in dataset:
            keys = tf.reshape(batch[self._key], (-1, tf.shape(batch[self._key])[-1]))
            values = tf.reshape(
                batch[self._value], (-1, tf.shape(batch[self._value])[-1])
            )
            for i in range(tf.shape(keys)[0]):
                d[keys[i, 0].numpy()] = values[i].numpy()

        # Build faiss index.
        logger.info("Building image index...")
        labels = np.array(list(d.keys()))
        data = np.stack(list(d.values()))
        db = faiss.IndexFlatL2(self._dim)
        db.add(data)

        self._labels = labels
        self._db = db

    def get_url(self, index: int):
        raise NotImplementedError

    def search(self, query, k=1):
        if not isinstance(query, np.ndarray):
            query = np.array([query], dtype=np.float32)

        _, index = self._db.search(query, k)
        urls = [self.get_url(i) for i in index[0].tolist()]
        if k == 1:
            return urls[0]
        return urls


class ImageRetriever(_Retriever):
    """Image retriever for visualization."""

    def __init__(
        self,
        path: Path,
        key: str = "image_hash",
        value: str = "image_embedding",
        condition: Dict[str, Any] = None,
        image_path: Path = None,
        dim: int = 512,
        **kwargs,
    ):
        super().__init__(path, key, value, condition, dim)
        self._dataspec = DataSpec("crello-images", path, **kwargs)
        if self._condition is None:
            self._condition = {
                "key": "type",
                "values": ("imageElement", "maskElement", "svgElement"),
            }
        self._image_path = image_path or self._path / "images"

    def get_url(self, index: int):
        label = self._labels[index]
        if label:
            return make_data_uri(self._image_path / (label.decode() + ".png"))
        return ""


class TextRetriever(_Retriever):
    """Text retriever for visualization."""

    def __init__(
        self,
        path: Path,
        key: str = "text_hash",
        value: str = "text_embedding",
        condition: Dict[str, Any] = None,
        text_path: Path = None,
        dim: int = 512,
        **kwargs,
    ):
        super().__init__(path, key, value, condition, dim)
        self._dataspec = DataSpec("crello-texts", path, **kwargs)
        if self._condition is None:
            self._condition = {
                "key": "type",
                "values": ("textElement",),
            }
        self._text_path = text_path or self._path / "texts"

    def get_url(self, index: int):
        label = self._labels[index]
        if label:
            url = self._text_path / (label.decode() + ".txt")
            with tf.io.gfile.GFile(str(url), "rb") as f:
                text = f.read()
            return text.decode()
        return ""


def make_data_uri(url: Union[str, Path], mime_type="image/png"):
    if isinstance(url, Path):
        url = str(url)
    with tf.io.gfile.GFile(url, "rb") as f:
        image_bytes = f.read()
    data = b64encode(image_bytes).decode("ascii")
    return "data:%s;base64,%s" % (mime_type, data)
