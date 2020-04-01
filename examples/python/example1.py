import numpy as np

from .utility.helper import get_np_buffer_ptr
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef

def main():
    a = np.random.randn(5, 5, 3).astype(np.float32)
    print(a)
    print("Call by Ref")
    a_ref = Matrix3fRef(a)


    print(get_np_buffer_ptr(a))
    b_ref = opf.kernel.laplacian(a_ref, iterations=1)
    print("Result")
    print(np.asarray(b_ref))

    print("Call by Copy")

    a_cp = Matrix3f(a)

    b_cp = opf.kernel.laplacian(a_cp)

    print("Result")
    print(np.asarray(b_cp))

# def main():
#     a = np.arange(5*5*3).reshape((5,5,3)).astype(np.float32)
#     print(a)
#     print("Call by Ref")
#     a_ref = Matrix3fRef(a)


#     print(get_np_buffer_ptr(a))
#     b_ref = opf.kernel.laplacian(a_ref, iterations=1)
#     print("Result")
#     print(np.asarray(b_ref))

#     print("Call by Copy")

#     a_cp = Matrix3f(a)

#     b_cp = opf.kernel.laplacian(a_cp)

#     print("Result")
#     print(np.asarray(b_cp))


if __name__ == "__main__":
    main()