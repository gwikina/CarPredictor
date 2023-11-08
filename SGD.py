import numpy as np

class SGD():
    def __init__(self, batch_size=1, lr=1e-3, max_epoch=100):
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch
        self.X = None
        self.y = None
        self.w = None

    def loss(self, x, y):
        return (x.dot(self.w)-y)**2

    def loss_grad(self, x, y):
        return 2*(x.dot(self.w)-y)

    def batch_loss(self, X, y):
        total_loss = 0
        for i in range(len(X)):
            total_loss += self.loss(X[i, :], y[i])
        return total_loss / len(X)

    def batch_loss_grad(self, X, y):
        total_loss_grad = np.zeros(X.shape[1])
        for i in range(len(X)):
            total_loss_grad += self.loss_grad(X[i, :], y[i])
        return total_loss_grad / len(X)

    def fit(self, X, y):
        self.X = np.hstack((np.ones((len(X), 1)), X.values))
        self.y = y.values
        self.w = np.random.rand(self.X.shape[1])
        
        for epoch in range(self.max_epoch):
            if self.batch_size == 0:
                self.w -= self.lr * self.batch_loss_grad(self.X, self.y)
            else:
                for i in range(0, len(self.X), self.batch_size):
                    batch_X = self.X[i:i + self.batch_size]
                    batch_y = self.y[i:i + self.batch_size]
                    self.w -= self.lr * self.batch_loss_grad(batch_X, batch_y)
