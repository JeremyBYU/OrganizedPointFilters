import numpy as np
import organizedpointfilters as opf
from organizedpointfilters import Matrix3f, Matrix3fRef


def get_np_buffer_ptr(a):
    pointer, read_only_flag = a.__array_interface__['data']
    return hex(pointer)

def main():
    a = np.arange(5*5*3).reshape((5,5,3)).astype(np.float32)
    print("By Ref")
    print(a)
    print(a.shape)
    print(get_np_buffer_ptr(a))

    b = Matrix3fRef(a)
    print(b)
    c = np.asarray(b)
    print(c)
    print(c.shape)
    print(get_np_buffer_ptr(c))


    print("Check with Copy!!")

    b = Matrix3f(a)
    print(b)
    c = np.asarray(b)
    print(c)
    print(c.shape)
    print(get_np_buffer_ptr(c))
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()