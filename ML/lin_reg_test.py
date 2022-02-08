import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt

X,y=datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
print(X)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

print(X.shape)
print(y.shape)
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="b", s=30)
plt.show()


from lin_reg import LinearRegression
regr=LinearRegression()
regr.fit(X_train, y_train)
#
y_predict=regr.predict(X_test)
#
#
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted) ** 2)
# #
mse_value=mse(y_test, y_predict)
print(mse_value)
# #
# #
y_pred_line=regr.predict(X)
cmap=plt.get_cmap('viridis')
fig=plt.figure(figsize=(8,8))
m1=plt.scatter(X_train, y_train, color=cmap(0.9), s=20)
m2=plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.legend()

plt.show()
