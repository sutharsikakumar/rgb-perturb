import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# load image
image_path = '/Users/sutharsikakumar/Desktop/55.jpg'  
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2d list
pixels = image.reshape(-1, 3)

# fixed colors for final figure
fixed_colors = np.array([
    [231, 76, 60],   # Red
    [46, 204, 113],  # Green
    [52, 152, 219],  # Blue
    [155, 89, 182],  # Purple
    [241, 196, 15],  # Yellow
]) / 255.0 

# add perturbation
def perturb_and_cluster(channel_index, perturb_range=(0, 255), num_clusters=5):
    perturbed_pixels = pixels.copy()
    # Add random perturbation to the specified channel
    perturbed_pixels[:, channel_index] += np.random.randint(*perturb_range, size=pixels.shape[0]).astype(np.uint8)
    perturbed_pixels = np.clip(perturbed_pixels, 0, 255)

    # k-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(perturbed_pixels)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # centroids
    centroid_brightness = np.mean(centroids, axis=1)
    brightness_order = np.argsort(centroid_brightness)
    
    # sorted labels mapping
    label_map = {old: new for new, old in enumerate(brightness_order)}
    new_labels = np.array([label_map[label] for label in labels])
    
    # segment image
    segmented_img = fixed_colors[new_labels].reshape(image.shape)
    return segmented_img, centroids[brightness_order], new_labels

# cluster
perturb_ranges = (0, 100)  # Example perturbation range
segmented_red, centroids_red, labels_red = perturb_and_cluster(0, perturb_ranges)
segmented_green, centroids_green, labels_green = perturb_and_cluster(1, perturb_ranges)
segmented_blue, centroids_blue, labels_blue = perturb_and_cluster(2, perturb_ranges)

# plot
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Perturbed Red Channel')
plt.imshow(segmented_red)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title('Perturbed Green Channel')
plt.imshow(segmented_green)
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Perturbed Blue Channel')
plt.imshow(segmented_blue)
plt.axis('off')

plt.tight_layout()
plt.show()

# centroids plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# centroids plot
for centroids, title, marker in [(centroids_red, 'Red Channel', 'o'), 
                                (centroids_green, 'Green Channel', '^'), 
                                (centroids_blue, 'Blue Channel', 's')]:
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
              marker=marker, label=f'Centroids ({title})', alpha=0.8)

ax.set_title('Centroids for Perturbed Channels')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')
ax.legend()
plt.show()