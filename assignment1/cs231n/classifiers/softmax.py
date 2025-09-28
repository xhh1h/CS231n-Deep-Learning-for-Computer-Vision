from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss += logp[y[i]]


        # 计算梯度
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j] - 1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]
        # 归一化
        loss /= num_train
        dW /= num_train

        # 加正则化
        loss += reg * np.sum(W**2)
        dW += 2 *reg * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs:
    - W: (D, C) 权重矩阵
    - X: (N, D) 数据矩阵
    - y: (N,) 标签向量
    - reg: 正则化强度

    Returns:
    - loss: float
    - dW: (D, C) 权重梯度
    """
    num_train = X.shape[0]

    # (N, C) 计算分数
    scores = X.dot(W)

    # 数值稳定性
    scores -= np.max(scores, axis=1, keepdims=True)

    # (N, C) softmax 概率
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算 loss
    correct_class_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_class_log_probs) / num_train
    loss += reg * np.sum(W * W)

    # 计算梯度
    dscores = probs.copy()
    dscores[np.arange(num_train), y] -= 1   # (N, C)
    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W

    return loss, dW

