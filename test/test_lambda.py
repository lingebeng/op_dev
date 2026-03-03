from functools import partial

def fn(activation):
    print(activation)

activations = ["relu", "gelu", "swish"]
kernels = []

for act in activations:
    # Python 的 lambda 是延迟绑定的，它记住的是 act 这个变量的引用，而不是当时的值。
    wrapped_kernel = lambda:fn(activation=act)
    kernels.append(wrapped_kernel)

kernels[0]()
# 输出: swish
kernels[1]()
# 输出: swish
kernels[2]()
# 输出: swish