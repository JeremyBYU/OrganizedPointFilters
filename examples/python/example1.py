import numpy as np
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

def get_np_buffer_ptr(a):
    pointer, read_only_flag = a.__array_interface__['data']
    return hex(pointer)

def main():
    a = np.arange(5*5*3).reshape((5,5,3)).astype(np.float32)
    b = np.ones((5, 5, 3), dtype=np.float32)

    print("Call by Ref")
    a_ref = Matrix3fRef(a)
    b_ref = Matrix3fRef(b)

    print(get_np_buffer_ptr(a))
    c = opf.kernel.laplacian(a_ref, b_ref,  3)

    print("Call by Copy")

    a_cp = Matrix3f(a)
    b_cp = Matrix3f(b)

    d = opf.kernel.laplacian(a_cp, b_cp,  3)


if __name__ == "__main__":
    main()