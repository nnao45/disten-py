"""
Ref.
      [1] Peng Li, Chengyu Liu, Ke Li, Dingchang Zheng, Changchun Liu,
          Yinglong Hou. "Assessing the complexity of short-term heartbeat
          interval series by distribution entropy", Med Biol Eng Comput,
          2015, 53: 77-87.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                     (C) Peng Li 2013-2017
If you use the code, please make sure that you cite reference [1]
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import numpy as np
import pandas as pd
from scipy.linalg import hankel
from scipy.spatial.distance import pdist
from typing import List
import sys
import math

def disten(ser: List[float], m: int = 2, tau: int = 8 , B: int = 512) -> float:
    """
    @param ser: time-series (vector in a column)
    @param m: embedding dimension (scalar)
    @param tau: time delay (scalar)
    @param B: bin number for histogram (scalar)
    """

    # rescaling
    rescaled = [y / (max(ser) - min(ser) + sys.float_info.epsilon) for y in [x - min(ser) for x in ser]]

    # distance matrix
    N = len(rescaled) - (m - 1) * tau
    if N < 0:
        raise(f"ser is too short: {len(ser)}")
    ind = hankel(np.arange(1, N+1), np.arange(N, len(rescaled)+1))
    rnt = [[rescaled[z-1] for z in y] for y in [x[::tau] for x in ind]]
    dv = pdist(rnt, 'chebychev')

    # esimating probability density by histogram
    num = pd.cut(dv, np.linspace(0, 1, B), include_lowest=True).value_counts().to_numpy()
    freq = [x / num.sum() for x in num]

    # disten calculation
    prepared = [math.log2(y) for y in [x + sys.float_info.epsilon for x in freq]]
    return -sum([x * y for (x, y) in zip(prepared, freq)]) / math.log2(B)
