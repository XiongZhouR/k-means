#!/usr/bin/python
# coding:utf-8

import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # 忽略warnings


def calc_dist(vec1, vec2):
    '''计算两个向量的欧氏距离'''
    return np.sqrt(np.sum((vec2-vec1)**2))


def rand_cent(dataSet, k):
    """取k个随机质心"""
    m, n = dataSet.shape
    # k个质心,列数跟样本的列数一样
    centroids = np.zeros((k, n))
    # 随机选出k个质心
    for i in range(k):
        index = int(np.random.uniform(0, m))
        centroids[i, :] = dataSet[index, :]
    return centroids


def _SSE_(dataSet, k, dist_eval=calc_dist, create_cent=rand_cent,cycle_num=500):
    '''
    计算误差的多少，通过这个方法来确定 k 为多少比较合适，这个其实就是一个简化版的 kMeans
    :param dataSet: 数据集
    :param k: 簇的数目
    :param dist_eval: 计算距离
    :param create_cent: 创建初始质心
    :param cycle_num：默认迭代次数
    :return:
    '''
    m = dataSet.shape[0]  # 计算样本个数
    # 样本的属性,第一列保存该样本属于哪个簇,第二列保存该样本与所属簇的误差
    cluster_Assign = np.array(np.zeros((m, 2)))
    centroids = rand_cent(dataSet, k)  # 随机创建质心

    while cycle_num > 0:
        # 循环每一个样本,更新每一个样本所在的簇
        for i in range(m):
            min_dist = np.inf  # inf为infinity无限大
            min_index = 0  # 定义样本所属的簇
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                dist = dist_eval(centroids[j, :], dataSet[i, :])
                if dist < min_dist:
                    min_dist = dist  # 更新最小距离
                    #cluster_Assign[i, 1] = min_dist
                    min_index = j  # 更新样本所属的簇
            cluster_Assign[i,:] = min_index,min_dist
            cycle_num-=1

        # 更新质心
        for j in range(k):
            cluster_Index = np.nonzero(cluster_Assign[:, 0] == j)
            sample_InCluster = dataSet[cluster_Index]
            # 计算质心
            centroids[j, :] = np.mean(sample_InCluster, axis=0)
    return np.mat(cluster_Assign[:,1].sum(0))[0,0]

# 画图展示手肘法确定的k
df = pd.read_csv('testset3.csv')
data = df.values

elbow_k=[]
for i in range(2,10):
    _SSE_1=_SSE_(data,i)
    elbow_k.append([i,_SSE_1])
elbow_k=np.matrix(elbow_k)

plt.plot(elbow_k[:,0],elbow_k[:,1],c='b')
plt.title("elbow_k")
plt.xlabel("k")
plt.ylabel("SSE")
plt.show()

