# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from PCA import readfile

(origin, next, Ope1) = (readfile())
w, v = LA.eig(Ope1)
s = sorted(w)
vectors = []

print(w)
w = list(w)
print(s)
for i in range(10):
    print(s[-(i + 1)], v[w.index(s[-(i + 1)])])
w = list(w)
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
x = []
y = []
z = []
for i in range(1):
    
    data = np.array(v)[w.index(s[-(i+1)])].reshape(3, 36)
    print(data)
    x = x + list(data[0])
    y = y + list(data[0])
    z = z + list(data[0])
ax.scatter(x[:], y[:], z[:], c='y')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()