import numpy as np
import matplotlib.pyplot as plt
import math

from astropy.coordinates import distances


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KMeans_clusters:
    def __init__(self, k = 10, max_iter = 100,random_seed = 37):
        self.k = k
        self.max_iter =max_iter
        self.random_seed =random_seed
        self.centroids = None  # 聚类中心
        self.clusters = None # output

    # 初始化聚类
    def initialize_centroids(self,X):
        np.random.seed(self.random_seed)
        self.centroids = X[np.random.choice(X[0].shape,self.k,replace=False)]
        return self.centroids

    def coumpute_distance(self,X):
        distances = np.zeros((X[0].shape, self.k))
        for i in range(self.k):
            distances[:,i] = np.apply_along_axis(euclidean_distance,1,X,self.centroids[i])

        return distances

    def assign_centroids(self,distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self,X):
        centroids = np.zeros((self.k,X.shape[1]))

        for i in range(self.k):
            centroids[i] = np.mean(X[self.centroids ==i],axis=0)

        return centroids

    def is_converged(self,old_centroids,new_centroids):
        distance   = np.sqrt(np.sum((old_centroids-new_centroids)**2))

        return distance ==0

    def fit(self,X):
        self.centroids = self.initialize_centroids(X)

        for i in range(self.max_iter):
            distances = self.coumpute_distance(X)
            self.clusters = self.assign_centroids(distances)
            old_centroids = self.centroids
            self.centroids = self.update_centroids(X)

            if self.is_converged(old_centroids, self.centroids):
                break

    def predict(self,X):
        # 计算每个样本到每个聚类中心的距离
        distances = self.compute_distances(X)
        # 分配每个样本到最近的聚类中心
        clusters = self.assign_clusters(distances)
        # 返回预测的聚类结果
        return clusters






