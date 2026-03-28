import numpy as np
import pandas as pd
#激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#sigmoid函数的导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

#权重
##隐藏层权重
w1=np.random.rand(2,4)
b1=np.random.rand(1,4)
##输出层权重
w2=np.random.rand(4,1)
b2=np.random.rand(1,1)

#训练数据
df = pd.read_csv("gender_height_weight_100.csv")
X = df[["height_cm", "weight_kg"]].values
y = df[["label_num"]].values

#标准化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

#训练10000次
for epoch in range(10000):
    #前向传播
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    #计算损失
    loss = np.mean((a2 - y) ** 2)

    #反向传播
    d_a2 = 2 * (a2 - y) / y.size
    d_z2 = d_a2 * sigmoid_derivative(z2)
    d_w2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = np.dot(d_z2, w2.T)
    d_z1 = d_a1 * sigmoid_derivative(z1)
    d_w1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    #更新权重和偏置
    learning_rate = 0.01
    w1 -= learning_rate * d_w1
    b1 -= learning_rate * d_b1
    w2 -= learning_rate * d_w2
    b2 -= learning_rate * d_b2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")