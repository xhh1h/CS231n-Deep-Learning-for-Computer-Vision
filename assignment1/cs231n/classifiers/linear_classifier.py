from __future__ import print_function

import os
import numpy as np
from builtins import range, object
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            ###################################################################
            # 1. 随机采样 batch_size 个训练样本
            ###################################################################
            batch_idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            ###################################################################
            # 2. 计算当前 batch 的 loss 和梯度
            ###################################################################
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            ###################################################################
            # 3. 参数更新
            ###################################################################
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels.
        """
        scores = X.dot(self.W)           # (N, C)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        """
        raise NotImplementedError

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = {"W": self.W}
        np.save(fpath, params)
        print(fname, "saved.")
    
    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.W = params["W"]
            print(fname, "loaded.")
            return True


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
