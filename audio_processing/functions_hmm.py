import numpy as np

def elog(x):
    res = np.log(x, where=(x!=0))
    res[np.where(x==0)] = -(10.0**8)
    return (res)

def logSumExp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return (x_max + np.log(sumexp))


def exp_normalize(x, axis=None, keepdims=False):
    b = x.max(axis=axis, keepdims=keepdims)
    y = np.exp(x - b)
    return y / y.sum(axis=axis, keepdims=keepdims)

