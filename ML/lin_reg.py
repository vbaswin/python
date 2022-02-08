import numpy as np    
# x=np.array([[1,2],[3,4],[5,6]])
# print(x.shape)
# x=np.zeros(5)
# print(x)

class LinearRegression:
 def __init__(self, lr=0.01, n_iters=1000): # constructor for initialising
   # instance variable or object variable . lr - learning rate , n_iters -
   # no of iterations 
   self.lr = lr
   self.n_iters = n_iters
   self.weights = None  #Θ₁
   self.bias = None #Θ₀

 def fit(self, X, y):
     # init parameters
     n_samples, n_features = X.shape
     self.weights = np.zeros(n_features)
     self.bias = 0
#
     for _ in range(self.n_iters):
         y_predicted = np.dot(X, self.weights) + self.bias #h_theta_x(i)
         d_theta_one = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
         d_theta_not = (1 / n_samples) * np.sum((y_predicted - y))

         self.weights -= self.lr * d_theta_one
         self.bias -= self.lr * d_theta_not

 def predict(self, X):
     y_predicted = np.dot(X, self.weights) + self.bias
     return y_predicted