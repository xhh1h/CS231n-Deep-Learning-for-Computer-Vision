from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer, of shape (N, d1, ..., dk)
    - w: Weights for the affine layer, of shape (D, M)
    - b: Biases for the affine layer, of shape (M,)

    Returns a tuple of:
    - out: Output from the ReLU, of shape (N, M)
    - cache: (fc_cache, relu_cache) for backward pass
    """
    # 1. 全连接前向传播
    a, fc_cache = affine_forward(x, w, b)

    # 2. ReLU 前向传播
    out, relu_cache = relu_forward(a)

    # 3. 保存 cache
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: (fc_cache, relu_cache) from affine_relu_forward

    Returns a tuple of:
    - dx: Gradient with respect to input x, shape (N, d1, ..., dk)
    - dw: Gradient with respect to weights w, shape (D, M)
    - db: Gradient with respect to biases b, shape (M,)
    """
    fc_cache, relu_cache = cache

    # 1. ReLU 反向传播
    da = relu_backward(dout, relu_cache)

    # 2. 全连接反向传播
    dx, dw, db = affine_backward(da, fc_cache)

    return dx, dw, db

