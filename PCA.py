# -*- coding: utf-8 -*-

from __future__ import print_function
import scipy.io as sio
import sys, os, math
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np

# 将一个list类型的数据转制并返回list
def listtranspose(alist):
    temp = np.array(np.matrix(alist).transpose())
    result = []
    for i in temp:
        result.append(list(i))
    return result

#读取文件并预处理
def readfile():
    input = sio.loadmat('pcatrainjoints.mat')
    samples = input.get('joints')
    origin = []

    ind = 0
    for i in samples:
        origin.append(list(i.reshape(-1)))
        for j in range(3, len(origin[ind])):
            #origin[ind][j] = (origin[ind][j] - origin[ind][j % 3]) / 300
            # 归一化
            origin[ind][j] = origin[ind][j] / 300

            #for j in range(3):
            #    origin[ind][j] = 0
            #ind = ind + 1

    # 归一化之后的数据，需要传出去做最后重构误差分析
    goback = origin

    # 将origin转制（转制后每个样本呈纵向排列）
    origin = listtranspose(origin)

    #减去平均值
    global sumadjust
    sumadjust = []
    ind = 0
    for i in origin:
        ave = sum(i) / len(i)
        sumadjust.append(ave)
        for j in range(len(i)):
            i[j] = i[j] - ave
        origin[ind] = i
        ind = ind + 1

    # 计算协方差
    Ope1 = np.matrix(np.cov(origin))
    return goback, origin, Ope1


# 计算不同数量的特征向量重构后的矩阵： reorigin
def notmain(vecnum, origin, Ope1):

    # 这里的origin已经转制过了

    # 特征
    w, v = LA.eig(Ope1)
    w = list(w)
    s = sorted(w)
    vectors = []
    for i in range(vecnum):
        vectors.append(list(np.array(v)[w.index(s[-(i+1)])])) 
    # 选取的特征向量组
    eigvec = np.matrix(vectors)

    orimat = np.matrix(origin) # Origin as matrix

    finalresult = np.dot(eigvec, orimat) # 最终结果
    reorigin = np.dot(eigvec.I, finalresult) # 重构
    ind = 0

    # 恢复之前减去的平均值
    for i in reorigin:
        for j in range(len(i)):
            i[j] = i[j] + sumadjust[ind]
        reorigin[ind] = i
        ind = ind + 1


    return reorigin.T






# main
(origin, next, Ope1) = readfile()
x = []
y = []
for i in range(1, 108):
    temp = notmain(i, next, Ope1)
    sum = 0
    
    for j in range(0, 72757, 10):
       for k in range(108):
           sum = sum + math.fabs(temp[j, k] - origin[j][k])
    y.append(sum / (7275 * 108))
    x.append(i)
    print(i)
    
fig = plt.figure() 
ax1 = fig.add_subplot(111) 
#设置标题 
ax1.set_title('Scatter Plot') 
#设置X轴标签 
plt.xlabel('X') 
#设置Y轴标签 
plt.ylabel('Y') 
#画散点图 
ax1.scatter(x,y,c = 'r',marker = 'o') 
#设置图标 
plt.legend('x1') 
#显示所画的图 
plt.show()