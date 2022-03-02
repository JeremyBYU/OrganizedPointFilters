from os import path
import numpy as np
from pathlib import Path
from datetime import date
import pytest

import organizedpointfilters as opf
from organizedpointfilters import Matrix3fRef

def test_cpu_bilateral_k3_pcd1(loops=2):
    """
    Just making sure it doesn't crash
    """
    opc = np.ones((20, 20, 3))
    a = Matrix3fRef(opc)
    b = opf.filter.bilateral_K3(a, loops)