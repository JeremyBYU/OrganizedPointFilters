from os import path
import numpy as np
import sys
from examples.python.utility.helper import (load_pcd_file)
import organizedpointfilters as opf
import organizedpointfilters.cuda as opf_cuda
from organizedpointfilters import Matrix3fRef
import ipdb

THIS_DIR = path.dirname(path.realpath(__file__))
PCD_DIR = path.join(THIS_DIR, '../', '../', 'fixtures', 'pcd')

pc, pc_image = load_pcd_file(path.join(PCD_DIR, 'pc_01.pcd'), stride=1)
pc, pc_image = load_pcd_file(path.join(PCD_DIR, 'pc_02.pcd'), stride=2) # illegal access if stride is 1. Updating cupy to >10 made  this error appear.
# looks like 430 X 430 is the limit
opc = np.ascontiguousarray(pc_image[:, :, :3].astype('f4'))



print(opc.shape, opc.dtype, opc.flags)


def laplacian_cuda_opc(opc, loops):
    b = opf_cuda.kernel.laplacian_K3_cuda(opc, loops, 1.0)
    return b

laplacian_cuda_opc(opc, 2)
# bilateral_cuda_opc(opc, 2)