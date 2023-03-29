import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def get_svg_size(input_path: Path) -> Tuple[int, int]:
    svg_root = ET.parse(input_path).getroot()
    canvas_width = math.ceil(float(svg_root.get("width")))
    canvas_height = math.ceil(float(svg_root.get("height")))
    return (canvas_width, canvas_height)


class Rasterizer:
    def __init__(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--hide-scrollbars")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.options = options

    def __call__(self, svg_path: Path, svg_img_path: Path, size: List[int]):
        assert len(size) == 2
        url = f"file://{str(svg_path.absolute())}"  # need full path
        driver = webdriver.Chrome(options=self.options)
        driver.set_window_size(*size)
        driver.get(url)
        driver.get_screenshot_as_file(str(svg_img_path))
        driver.quit()
