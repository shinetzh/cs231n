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
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in np.arange(num_train):
        scores = X[i].dot(W)
        scores = np.exp(scores)
        correct_score = scores[y[i]]
        sum_score = np.sum(scores)
        loss = loss-np.log(correct_score/sum_score)
        for j in np.arange(num_class):
            if j == y[i]:
                dW[:,j] += (scores[j]/sum_score-1)*X[i]
            else:
                dW[:,j] += scores[j]/sum_score*X[i]
  loss = 1/num_train*loss+0.5*reg*np.sum(W*W)
  dW = 1/num_train*dW+reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  scores = scores-np.max(scores,axis = 1).reshape(N,-1)
  scores = np.exp(scores)
  correct_scores = scores[np.arange(num_train),y]
  sum_rows = np.sum(scores,axis = 1)
  probabilities = 1/sum_rows*correct_scores
  loss = 1/num_train*np.sum(-np.log(probabilities))+0.5*reg*np.sum(W*W)
  
  mask = scores
  sum_rows = np.tile(sum_rows,(num_class,1)).T
  mask = 1/sum_rows*mask
  mask[np.arange(num_train),y] -= 1
  dW = 1/num_train*np.dot(X.T,mask)+reg*W
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

