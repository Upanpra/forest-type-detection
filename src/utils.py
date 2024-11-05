# taken from https://github.com/pytorch/vision/blob/72686211e2a8b78e5a5dc8c28be34eb9cfcdad4c/torchvision/utils.py#L565

import collections
from itertools import repeat
from typing import Any, Tuple


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8
    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))
