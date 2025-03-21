import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


image_path = 'FILE-PATH'  
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pixels = image.reshape(-1, 3)


num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pixels)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


segmented_img = centroids[labels].reshape(image.shape).astype(np.uint8)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(segmented_img)
plt.axis('off')

plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


colors = plt.cm.get_cmap('tab10', num_clusters)

for i in range(num_clusters):
    cluster_points = pixels[labels == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
               color=colors(i), label=f'Cluster {i}', alpha=0.6)


ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
           s=300, c='black', marker='X', label='Centroids')

ax.set_title('K-means Clustering on Image Pixels in RGB Space')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.legend()
plt.show()