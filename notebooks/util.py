import itertools
import sys
from typing import Dict

sys.path.append("../src/mfp")
from mfp.models.mfp import MFP


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def load_model(path: str, input_columns: Dict):
    model = MFP(
        input_columns,
        latent_dim=256,
        num_blocks=4,
        block_type="deepsvg",
        masking_method="random",
    )
    model.compile(optimizer="adam")
    best_or_final = "best"
    model.load_weights(f"{path}/{best_or_final}.ckpt")
    return model
