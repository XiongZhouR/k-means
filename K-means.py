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


def Kmeans(dataSet, k, dist_eval=calc_dist, create_cent=rand_cent):
    """
    创建随机的K个点作为起始质心
    当任意一个点的簇分配结果发生改变时：
        对数据中的每个数据点：
            对每个质心:
                计算质心与数据点之间的距离
            将数据点分配到距其最近的簇
        对每个簇，计算簇中所有点的均值并将其作为质心
    """
    m = dataSet.shape[0]  # 计算样本个数
    # 样本的属性,第一列保存该样本属于哪个簇,第二列保存该样本与所属簇的误差
    cluster_Assign = np.array(np.zeros((m, 2)))
    centroids = rand_cent(dataSet, k)  # 随机创建质心

    cluster_change = True  # 程序中止条件
    while cluster_change:
        cluster_change = False

        # 循环每一个样本,更新每一个样本所在的簇
        for i in range(m):
            min_dist = np.inf  # inf为infinity无限大
            min_index = 0  # 定义样本所属的簇
            # 循环计算每一个质心与该样本的距离
            for j in range(k):
                dist = dist_eval(centroids[j, :], dataSet[i, :])
                if dist < min_dist:
                    min_dist = dist  # 更新最小距离
                    cluster_Assign[i, 1] = min_dist
                    min_index = j  # 更新样本所属的簇
            # 如果样本所属的簇发生了变化
            if cluster_Assign[i, 0] != min_index:
                cluster_change = True  # 质心要重新计算
                cluster_Assign[i, 0] = min_index  # 更新样本所属的簇

        # 更新质心
        for j in range(k):
            cluster_Index = np.nonzero(cluster_Assign[:, 0] == j)
            sample_InCluster = dataSet[cluster_Index]
            # 计算质心
            centroids[j, :] = np.mean(sample_InCluster, axis=0)
    return centroids, cluster_Assign


def show_Cluster(dataSet, k, centroids, cluster_Assign):
    m, n = dataSet.shape
    if n != 2:
        print("聚类数据不是二维的!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大,聚类过多!")
        return 1

    # 画样本点
    for i in range(m):
        mark_Index = int(cluster_Assign[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[mark_Index])

    mark = ['+r', '+b', '+g', '+k', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()


df = pd.read_csv('testset3.csv')
data = df.values
k = 3
centriods, clusterassign = Kmeans(data, k)
show_Cluster(data, k, centriods, clusterassign)

#数据集网址:https://github.com/XiongZhouR/k-means.git

