"""Simple example showing how to pass pass an organized point clouds to organized point filters
"""
import numpy as np

from .utility.helper import get_np_buffer_ptr
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

def main():
    a = np.random.randn(5, 5, 3).astype(np.float32)
    a_ref = Matrix3fRef(a)
    print("Call by Ref (same memory buffer")
    print(get_np_buffer_ptr(a))
    print(get_np_buffer_ptr(np.asarray(a_ref)))
    print(a)
    b_ref = opf.filter.laplacian_K3(a_ref, iterations=1)
    print("Result")
    print(np.asarray(b_ref))


if __name__ == "__main__":
    main()