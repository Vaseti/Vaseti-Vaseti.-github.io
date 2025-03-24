import numpy as np
#输入层到隐藏层的权重矩阵
X=np.array([1.0,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1=np.array([0.1,0.2,0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)
print("----------------")
A1=np.dot(X,W1)+B1


#激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))
#隐藏层到第二层的权重矩阵
z1=sigmoid(A1)
print(A1)
print(z1)
print("----------------")
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])

print(z1.shape)
print(W2.shape)
print(B2.shape)
print("----------------")
A2=np.dot(z1,W2)+B2
z2=sigmoid(A2)

#激活函数
def identity_function(x):
    return x

#第二层到输出层的权重矩阵
W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(z2,W3)+B3
Y=identity_function(A3)
print(Y)