import numpy as np

input_dim = 4
out_dim = 3
h_dim = 5 #количество нейронов в первом слое

x = np.random.randn(input_dim) #входной вектор
w1 = np.random.randn(input_dim, h_dim) #матрица весов (1 слой)
b1 = np.random.randn(h_dim) #размерность вектора смещения

w2 = np.random.randn(h_dim, out_dim) #2 слой
b2 = np.random.randn(out_dim)

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def predict(x):
    t1 = x @ w1 + b1 #линейная часть
    h1 = relu(t1) #нелинейная часть
    t2 = h1 @ w2 + b2
    у = softmax(t2)
    return у

probs = predict(x)
pred_class = np.argmax(probs)
class_names = ['1', '2', '3']
print('Predicted class:', class_names[pred_class])
