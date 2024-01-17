# 导入必要的Python库
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN

# 读取图像并转换为NumPy数组
img = Image.open("input.png")
img_data = np.array(img)

# 计算像素数和特征数
rows, cols, channels = img_data.shape
num_pixels = rows * cols

# 将图像数据重塑为 (N, 3)形式
X = np.reshape(img_data, (num_pixels, channels))

# 创建一个DBSCAN聚类器，传入邻域半径和最小邻域点数
eps = 0.3 # 您可以根据需要修改这个值
min_samples = 3 # 您可以根据需要修改这个值
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# 训练聚类器，传入像素数据
dbscan.fit(X)

# 获取每个像素的簇标签，-1表示噪声点
labels = dbscan.labels_

# 获取簇的个数，不包括噪声点
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# 创建一个空的数组，用于存储每个簇的中心值
colors = np.zeros((n_clusters, channels))

# 遍历每个簇，计算其平均像素值
for i in range(n_clusters):
    # 获取当前簇中所有像素的索引
    indices = np.where(labels == i)[0]
    # 计算当前簇中所有像素的均值，作为中心值
    colors[i] = np.mean(X[indices], axis=0)

# 使用簇的中心值替换原来的像素值，得到压缩后的图像数据
new_X = colors[labels]
new_X = new_X.reshape((rows, cols, channels))

# 将新图像保存到磁盘
new_img = Image.fromarray(np.uint8(new_X))
new_img.save("DBSCAN.png") # 您可以根据需要修改这个文件名
