import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

FIXED_COLORS = np.array([
    [231, 76, 60],   
    [46, 204, 113], 
    [52, 152, 219], 
    [155, 89, 182],
    [241, 196, 15],  
]) / 255.0

def load_and_verify_image(image_path):
    """Load and verify image with error handling."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at: {image_path}")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def perform_kmeans(pixels, num_clusters=5):
    """Perform basic k-means clustering without perturbation."""
    pixels_float = pixels.astype(np.float32)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(pixels_float)
    
    centroids = kmeans.cluster_centers_
    centroid_brightness = np.mean(centroids, axis=1)
    brightness_order = np.argsort(centroid_brightness)
    
    label_map = {old: new for new, old in enumerate(brightness_order)}
    new_labels = np.array([label_map[label] for label in kmeans.labels_])
    
    return new_labels, centroids[brightness_order]

def perturb_and_cluster(pixels, channel_index, perturb_range=(0, 255), num_clusters=5):
    """Perform k-means clustering with perturbation on specified channel."""
    perturbed_pixels = pixels.astype(np.float32)
    perturbations = np.random.randint(*perturb_range, size=pixels.shape[0])
    perturbed_pixels[:, channel_index] = perturbed_pixels[:, channel_index] + perturbations
    perturbed_pixels = np.clip(perturbed_pixels, 0, 255)
    
    return perform_kmeans(perturbed_pixels, num_clusters)

def create_segmented_image(labels, original_shape):
    """Create segmented image using fixed colors."""
    segmented_pixels = FIXED_COLORS[labels]
    return segmented_pixels.reshape(original_shape)

def plot_confusion_matrices(image_path, perturb_range=(0, 100), num_clusters=5):
    try:
        image = load_and_verify_image(image_path)
        original_shape = image.shape
        pixels = image.reshape(-1, 3)
        
        base_labels, base_centers = perform_kmeans(pixels, num_clusters)
        base_segmented = create_segmented_image(base_labels, original_shape)
        
        red_labels, red_centers = perturb_and_cluster(pixels, 0, perturb_range, num_clusters)
        green_labels, green_centers = perturb_and_cluster(pixels, 1, perturb_range, num_clusters)
        blue_labels, blue_centers = perturb_and_cluster(pixels, 2, perturb_range, num_clusters)
        
        red_segmented = create_segmented_image(red_labels, original_shape)
        green_segmented = create_segmented_image(green_labels, original_shape)
        blue_segmented = create_segmented_image(blue_labels, original_shape)
        
        cm_red = confusion_matrix(base_labels, red_labels)
        cm_green = confusion_matrix(base_labels, green_labels)
        cm_blue = confusion_matrix(base_labels, blue_labels)
        
        cm_red_norm = cm_red.astype('float') / cm_red.sum(axis=1)[:, np.newaxis]
        cm_green_norm = cm_green.astype('float') / cm_green.sum(axis=1)[:, np.newaxis]
        cm_blue_norm = cm_blue.astype('float') / cm_blue.sum(axis=1)[:, np.newaxis]
        
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image)
        ax1.set_title('Original Image', pad=20)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(base_segmented)
        ax2.set_title('Base Segmentation', pad=20)
        ax2.axis('off')
        
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(red_segmented)
        ax4.set_title('Red Channel Perturbed', pad=20)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(green_segmented)
        ax5.set_title('Green Channel Perturbed', pad=20)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(blue_segmented)
        ax6.set_title('Blue Channel Perturbed', pad=20)
        ax6.axis('off')
        
        plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        def plot_cm(cm, ax, title):
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{title} Channel Perturbation')
            ax.set_xlabel('Perturbed Clusters')
            ax.set_ylabel('Original Clusters')
        
        plot_cm(cm_red_norm, axes[0], 'Red')
        plot_cm(cm_green_norm, axes[1], 'Green')
        plot_cm(cm_blue_norm, axes[2], 'Blue')
        
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for centroids, title, marker in [(red_centers, 'Red Channel', 'o'),
                                       (green_centers, 'Green Channel', '^'),
                                       (blue_centers, 'Blue Channel', 's')]:
            ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                      marker=marker, label=f'Centroids ({title})', alpha=0.8)
        
        ax.set_title('Centroids for Perturbed Channels')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.legend()
        plt.show()
        
        stability_metrics = {
            'red': np.diagonal(cm_red_norm).mean(),
            'green': np.diagonal(cm_green_norm).mean(),
            'blue': np.diagonal(cm_blue_norm).mean()
        }
        
        return stability_metrics
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the image file exists and the path is correct.")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check if the image file is valid and not corrupted.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    image_path = 'FILE-PATH'
    print(f"Attempting to process image: {image_path}")
    
    stability_metrics = plot_confusion_matrices(image_path, perturb_range=(0, 100), num_clusters=5)
    
    if stability_metrics:
        print("\nStability Metrics (average diagonal values):")
        for channel, stability in stability_metrics.items():
            print(f"{channel.capitalize()} channel stability: {stability:.3f}")

if __name__ == "__main__":
    main()