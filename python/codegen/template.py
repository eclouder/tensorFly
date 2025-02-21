import tvm
from tvm import te
class template(object):
    @staticmethod
    def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
