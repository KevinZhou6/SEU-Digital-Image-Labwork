# 导入必要的Python库
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


# 读取图像并转换为NumPy数组
img = Image.open("input.png")
img_data = np.array(img)

# 计算像素数和特征数
rows, cols, channels = img_data.shape
num_pixels = rows * cols

# 将图像数据重塑为 (N, 3)形式
X = np.reshape(img_data, (num_pixels, channels))

# 将图像数据聚类为k个颜色

k = 8 # 您可以根据需要修改这个值
kmeans = KMeans(n_clusters=k).fit(X)
labels = kmeans.predict(X)
colors = kmeans.cluster_centers_.astype(int )

# 使用聚类中心替换每个像素的颜色
new_X = colors[labels]
new_X = new_X.reshape((rows, cols, channels))

# 将新图像保存到磁盘
new_img = Image.fromarray(np.uint8(new_X))
new_img.save("K-Means_output.png") # 您可以根据需要修改这个文件名
