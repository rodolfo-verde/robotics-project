from collections import namedtuple
from dataclasses import dataclass
from typing import List

import numpy as np

DataTuple = namedtuple('DataTuple', 'key feats label')

@dataclass(unsafe_hash=True)
class HMMDataTuple:
    '''Class to store data for HMM training'''
    feats: np.ndarray
    states: List[int]