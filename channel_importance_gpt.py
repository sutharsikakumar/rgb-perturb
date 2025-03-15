import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import json
import re
from openai import OpenAI
from matplotlib.colors import LinearSegmentedColormap


FIXED_COLORS = np.array([
    [231, 76, 60],   # Red
    [46, 204, 113],  # Green
    [52, 152, 219],  # Blue
    [155, 89, 182],  # Purple
    [241, 196, 15],  # Yellow
    [52, 73, 94],    # Dark Blue
    [230, 126, 34],  # Orange
    [149, 165, 166], # Gray
]) / 255.0


def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                return {tuple(map(float, k.split(','))): v for k, v in cache_data.items()}
        except json.JSONDecodeError:
            print(f"Error reading {cache_file}. File may be corrupted.")
            return {}
    return {}

def save_cache(cache_file, cache_data):
    try:
        with open(cache_file, 'w') as f:
            cache_data_str_keys = {','.join(map(str, k)): v for k, v in cache_data.items()}
            json.dump(cache_data_str_keys, f)
    except Exception as e:
        print(f"Error writing to {cache_file}: {e}")

def load_and_verify_image(image_path):
    """Load and verify image with error handling."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_optimal_clusters_from_openai(distortions, client, cache_file='cache.json'):
    cache_data = load_cache(cache_file)
    cache_key = tuple(distortions)
    if cache_key in cache_data:
        print("Using cached result for distortions:", distortions)
        return cache_data[cache_key]
    
    pct_changes = [((distortions[i-1] - distortions[i]) / distortions[i-1]) * 100 
                   for i in range(1, len(distortions))]
    prompt = f"""Analyze these distortion values from K-means clustering and determine the optimal number of clusters using the Elbow Method.
Raw distortion values: {distortions}
Percentage changes: {[f'{x:.2f}%' for x in pct_changes]}
Rules: Optimal number must be between 2 and {len(distortions)}. Respond with only the number."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in computer vision and clustering."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        optimal_clusters = int(re.search(r'\b\d+\b', response.choices[0].message.content.strip()).group())
        if not (2 <= optimal_clusters <= len(distortions)):
            optimal_clusters = 5
        cache_data[cache_key] = optimal_clusters
        save_cache(cache_file, cache_data)
        return optimal_clusters
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return 5

def find_optimal_clusters(pixel_values, client, max_k=8):
    print("Finding optimal clusters...")
    distortions = [KMeans(n_clusters=k, random_state=0).fit(pixel_values).inertia_ 
                   for k in range(1, max_k)]
    optimal_k = get_optimal_clusters_from_openai(distortions, client)
    print(f"OpenAI suggests {optimal_k} clusters")
    return optimal_k, distortions

def perform_kmeans(pixels, num_clusters):
    pixels_float = pixels.astype(np.float32)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(pixels_float)
    centroids = kmeans.cluster_centers_
    centroid_brightness = np.mean(centroids, axis=1)
    brightness_order = np.argsort(centroid_brightness)
    label_map = {old: new for new, old in enumerate(brightness_order)}
    new_labels = np.array([label_map[label] for label in kmeans.labels_])
    return new_labels, centroids[brightness_order]

def perturb_and_cluster(pixels, channel_index, perturb_range=(0, 255), num_clusters=None):
    """Perform k-means clustering with perturbation on specified channel."""
    perturbed_pixels = pixels.astype(np.float32)
    perturbations = np.random.randint(*perturb_range, size=pixels.shape[0])
    perturbed_pixels[:, channel_index] = perturbed_pixels[:, channel_index] + perturbations
    perturbed_pixels = np.clip(perturbed_pixels, 0, 255)
    return perform_kmeans(perturbed_pixels, num_clusters)

def create_segmented_image(labels, original_shape):
    if len(FIXED_COLORS) < len(np.unique(labels)):
        raise ValueError("Not enough fixed colors for clusters")
    return FIXED_COLORS[labels].reshape(original_shape)

# add noise
def add_gaussian_noise(image, channel_index, noise_level):
    noisy_image = image.copy()
    noise = np.random.normal(0, noise_level, noisy_image.shape[:2])
    noisy_image[..., channel_index] = np.clip(noisy_image[..., channel_index] + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_uniform_noise(image, channel_index, noise_level):
    noisy_image = image.copy()
    noise = np.random.uniform(-noise_level, noise_level, noisy_image.shape[:2])
    noisy_image[..., channel_index] = np.clip(noisy_image[..., channel_index] + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image, channel_index, noise_density):
    noisy_image = image.copy()
    h, w = image.shape[:2]
    num_salt = int(np.ceil(noise_density * h * w))
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    noisy_image[salt_coords[0], salt_coords[1], channel_index] = 255
    num_pepper = int(np.ceil(noise_density * h * w))
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    noisy_image[pepper_coords[0], pepper_coords[1], channel_index] = 0
    return noisy_image

def channel_dropout(image, channel_index):
    dropout_image = image.copy()
    dropout_image[..., channel_index] = 0
    return dropout_image

# apply tests
def compute_image_similarity(original_image, perturbed_image):
    orig_img = original_image.astype(np.float32) / 255.0
    pert_img = perturbed_image.astype(np.float32) / 255.0
    ssim_value = np.mean([ssim(orig_img[..., i], pert_img[..., i], data_range=1.0) for i in range(3)])
    psnr_value = psnr(orig_img, pert_img, data_range=1.0) if np.any(orig_img != pert_img) else float('inf')
    return {"ssim": ssim_value, "psnr": psnr_value}

def evaluate_segmentation(original_labels, perturbed_labels):
    return adjusted_rand_score(original_labels, perturbed_labels)

def plot_confusion_matrices(image_path, client, perturb_range=(0, 100)):
    try:
        image = load_and_verify_image(image_path)
        pixels = image.reshape(-1, 3)
        optimal_clusters, _ = find_optimal_clusters(pixels, client)
        
        base_labels, base_centers = perform_kmeans(pixels, optimal_clusters)
        base_segmented = create_segmented_image(base_labels, image.shape)
        
        red_labels, red_centers = perturb_and_cluster(pixels, 0, perturb_range, optimal_clusters)
        green_labels, green_centers = perturb_and_cluster(pixels, 1, perturb_range, optimal_clusters)
        blue_labels, blue_centers = perturb_and_cluster(pixels, 2, perturb_range, optimal_clusters)
        
        red_segmented = create_segmented_image(red_labels, image.shape)
        green_segmented = create_segmented_image(green_labels, image.shape)
        blue_segmented = create_segmented_image(blue_labels, image.shape)
        
        cm_red = confusion_matrix(base_labels, red_labels)
        cm_green = confusion_matrix(base_labels, green_labels)
        cm_blue = confusion_matrix(base_labels, blue_labels)
        
        cm_red_norm = cm_red.astype('float') / cm_red.sum(axis=1)[:, np.newaxis]
        cm_green_norm = cm_green.astype('float') / cm_green.sum(axis=1)[:, np.newaxis]
        cm_blue_norm = cm_blue.astype('float') / cm_blue.sum(axis=1)[:, np.newaxis]
        
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)
        for ax, img, title in [
            (gs[0, 0], image, f'Original ({optimal_clusters} clusters)'),
            (gs[0, 1], base_segmented, 'Base Segmentation'),
            (gs[1, 0], red_segmented, 'Red Perturbed'),
            (gs[1, 1], green_segmented, 'Green Perturbed'),
            (gs[1, 2], blue_segmented, 'Blue Perturbed')
        ]:
            plt.subplot(ax)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for cm, ax, title in [
            (cm_red_norm, axes[0], 'Red'),
            (cm_green_norm, axes[1], 'Green'),
            (cm_blue_norm, axes[2], 'Blue')
        ]:
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title(f'{title} Perturbation')
            ax.set_xlabel('Perturbed Clusters')
            ax.set_ylabel('Original Clusters')
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for cents, title, marker in [
            (red_centers, 'Red', 'o'),
            (green_centers, 'Green', '^'),
            (blue_centers, 'Blue', 's')
        ]:
            ax.scatter(cents[:, 0], cents[:, 1], cents[:, 2], marker=marker, label=title, alpha=0.8)
        ax.set_title('Centroids for Perturbed Channels')
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.legend()
        plt.show()
        
        stability_metrics = {
            'red': np.nanmean(np.diagonal(cm_red_norm)),
            'green': np.nanmean(np.diagonal(cm_green_norm)),
            'blue': np.nanmean(np.diagonal(cm_blue_norm))
        }
        return stability_metrics
    except Exception as e:
        print(f"Error in confusion matrices: {e}")
        return None

# channel importance
def analyze_channel_degradation(image_path, client, noise_type="gaussian"):
    image = load_and_verify_image(image_path)
    pixels = image.reshape(-1, 3)
    optimal_clusters, _ = find_optimal_clusters(pixels, client)
    base_labels, _ = perform_kmeans(pixels, optimal_clusters)
    
    noise_levels = {
        "gaussian": [5, 10, 20, 30, 40, 50, 75, 100],
        "uniform": [10, 20, 40, 60, 80, 100, 125, 150],
        "salt_pepper": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    }[noise_type]
    noise_func = {"gaussian": add_gaussian_noise, "uniform": add_uniform_noise, 
                  "salt_pepper": add_salt_pepper_noise}[noise_type]
    
    channel_names = ["Red", "Green", "Blue"]
    ssim_results = {ch: [] for ch in channel_names}
    psnr_results = {ch: [] for ch in channel_names}
    seg_results = {ch: [] for ch in channel_names}
    sample_images = []
    
    for noise_level in noise_levels:
        for i, ch in enumerate(channel_names):
            noisy_image = noise_func(image, i, noise_level)
            if noise_level == noise_levels[3]:
                sample_images.append((ch, noisy_image))
            similarity = compute_image_similarity(image, noisy_image)
            noisy_labels, _ = perform_kmeans(noisy_image.reshape(-1, 3), optimal_clusters)
            ssim_results[ch].append(similarity["ssim"])
            psnr_results[ch].append(similarity["psnr"])
            seg_results[ch].append(evaluate_segmentation(base_labels, noisy_labels))
    
    fig = plt.figure(figsize=(15, 10))
    for i, (data, ylabel, title) in enumerate([
        (ssim_results, 'SSIM Score', 'SSIM'),
        (psnr_results, 'PSNR (dB)', 'PSNR'),
        (seg_results, 'Segmentation Similarity', 'Segmentation Stability')
    ], 1):
        plt.subplot(2, 2, i)
        for ch in channel_names:
            plt.plot(noise_levels, data[ch], marker='o', label=ch)
        plt.title(f'Channel-wise {title} - {noise_type.capitalize()}')
        plt.xlabel('Noise Level')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    for i, (ch, img) in enumerate([(None, image)] + sample_images[:3]):
        plt.subplot(2, 2, i+1)
        plt.imshow(img)
        plt.title('Original' if i == 0 else f'{ch} Perturbed')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return {"ssim": ssim_results, "psnr": psnr_results, "segmentation": seg_results}

def analyze_channel_dropout(image_path, client):
    image = load_and_verify_image(image_path)
    pixels = image.reshape(-1, 3)
    optimal_clusters, _ = find_optimal_clusters(pixels, client)
    base_labels, _ = perform_kmeans(pixels, optimal_clusters)
    
    channel_names = ["Red", "Green", "Blue"]
    ssim_values, psnr_values, seg_impact, dropout_images = [], [], [], []
    
    for i, ch in enumerate(channel_names):
        dropout_img = channel_dropout(image, i)
        dropout_images.append(dropout_img)
        similarity = compute_image_similarity(image, dropout_img)
        dropout_labels, _ = perform_kmeans(dropout_img.reshape(-1, 3), optimal_clusters)
        ssim_values.append(similarity["ssim"])
        psnr_values.append(similarity["psnr"])
        seg_impact.append(1 - evaluate_segmentation(base_labels, dropout_labels))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, data, title, ylabel in [
        (axes[0, 0], [1 - s for s in ssim_values], 'SSIM Drop', 'SSIM Drop'),
        (axes[0, 1], [(max(psnr_values) - p) / max(psnr_values) if max(psnr_values) > 0 else 0 for p in psnr_values], 
         'PSNR Drop', 'Normalized PSNR Drop'),
        (axes[1, 0], seg_impact, 'Segmentation Impact', 'Segmentation Change')
    ]:
        ax.bar(channel_names, data, color=['r', 'g', 'b'])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
    
    axes[1, 1].axis('off')
    for i, img in enumerate([image] + dropout_images):
        subax = fig.add_subplot(4, 3, 7 + i)
        subax.imshow(img)
        subax.set_title('Original' if i == 0 else f'No {channel_names[i-1]}')
        subax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return {"ssim_drop": [1 - s for s in ssim_values], "psnr_drop": [(max(psnr_values) - p) / max(psnr_values) 
            if max(psnr_values) > 0 else 0 for p in psnr_values], "segmentation_impact": seg_impact}

def generate_perturbation_heatmap(image_path, client, noise_type="gaussian"):
    image = load_and_verify_image(image_path)
    pixels = image.reshape(-1, 3)
    optimal_clusters, _ = find_optimal_clusters(pixels, client)
    base_labels, _ = perform_kmeans(pixels, optimal_clusters)
    
    noise_levels = {
        "gaussian": [0, 10, 25, 50, 75, 100],
        "uniform": [0, 20, 40, 80, 120, 160],
        "salt_pepper": [0, 0.02, 0.05, 0.1, 0.2, 0.3]
    }[noise_type]
    noise_func = {"gaussian": add_gaussian_noise, "uniform": add_uniform_noise, 
                  "salt_pepper": add_salt_pepper_noise}[noise_type]
    
    channel_names = ["Red", "Green", "Blue", "All"]
    ssim_heatmap = np.zeros((len(noise_levels), len(channel_names)))
    
    for i, level in enumerate(noise_levels):
        for j, ch in enumerate(channel_names[:3]):
            if level == 0:
                ssim_heatmap[i, j] = 1.0
            else:
                noisy_image = noise_func(image, j, level)
                ssim_heatmap[i, j] = compute_image_similarity(image, noisy_image)["ssim"]
        if level == 0:
            ssim_heatmap[i, 3] = 1.0
        else:
            all_noisy = image.copy()
            for k in range(3):
                all_noisy = noise_func(all_noisy, k, level)
            ssim_heatmap[i, 3] = compute_image_similarity(image, all_noisy)["ssim"]
    
    cmap = LinearSegmentedColormap.from_list("custom", [(0.8, 0, 0), (1, 1, 0), (0, 0.8, 0)], N=256)
    plt.figure(figsize=(12, 8))
    sns.heatmap(ssim_heatmap, annot=True, fmt=".3f", cmap=cmap, xticklabels=channel_names, 
                yticklabels=noise_levels, vmin=0, vmax=1)
    plt.title(f'Perturbation Heatmap - {noise_type.capitalize()} (SSIM)')
    plt.xlabel('Channel')
    plt.ylabel('Noise Level')
    plt.tight_layout()
    plt.show()
    
    return ssim_heatmap

def analyze_feature_attribution(image_path, client, noise_type="gaussian"):
    image = load_and_verify_image(image_path)
    pixels = image.reshape(-1, 3)
    optimal_clusters, _ = find_optimal_clusters(pixels, client)
    base_labels, base_centroids = perform_kmeans(pixels, optimal_clusters)
    
    noise_levels = {
        "gaussian": [0, 10, 25, 50, 75, 100],
        "uniform": [0, 20, 40, 80, 120, 160],
        "salt_pepper": [0, 0.02, 0.05, 0.1, 0.2, 0.3]
    }[noise_type]
    noise_func = {"gaussian": add_gaussian_noise, "uniform": add_uniform_noise, 
                  "salt_pepper": add_salt_pepper_noise}[noise_type]
    
    channel_names = ["Red", "Green", "Blue"]
    orig_importances = np.var(pixels, axis=0) / np.sum(np.var(pixels, axis=0))
    
    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2, 2, 1, polar=True)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist() + [0]
    values = orig_importances.tolist() + [orig_importances[0]]
    ax1.plot(angles, values, 'o-', label='Original')
    
    highest_noise = noise_levels[-1]
    for i, ch in enumerate(channel_names):
        noisy_image = noise_func(image, i, highest_noise)
        noisy_pixels = noisy_image.reshape(-1, 3)
        noisy_importances = np.var(noisy_pixels, axis=0) / np.sum(np.var(noisy_pixels, axis=0))
        values = noisy_importances.tolist() + [noisy_importances[0]]
        ax1.plot(angles, values, 'o-', label=f'{ch} Perturbed')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(channel_names)
    ax1.set_title('Feature Attribution Radar Chart')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    ax2 = plt.subplot(2, 2, 2)
    x = noise_levels[1:]
    y_data = {ch: [] for ch in channel_names}
    for level in x:
        for i, ch in enumerate(channel_names):
            noisy_image = noise_func(image, i, level)
            noisy_pixels = noisy_image.reshape(-1, 3)
            noisy_importances = np.var(noisy_pixels, axis=0) / np.sum(np.var(noisy_pixels, axis=0))
            y_data[ch].append(np.sum(np.abs(noisy_importances - orig_importances)))
    for ch, values in y_data.items():
        ax2.plot(x, values, 'o-', label=ch, color={'Red': 'r', 'Green': 'g', 'Blue': 'b'}[ch])
    ax2.set_title('Feature Attribution Difference')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Sum of Absolute Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    ax3.scatter(base_centroids[:, 0], base_centroids[:, 1], base_centroids[:, 2], c='k', marker='o', s=100, label='Original')
    for i, ch in enumerate(channel_names):
        noisy_image = noise_func(image, i, highest_noise)
        noisy_labels, noisy_centroids = perform_kmeans(noisy_image.reshape(-1, 3), optimal_clusters)
        ax3.scatter(noisy_centroids[:, 0], noisy_centroids[:, 1], noisy_centroids[:, 2], 
                    c=['r', 'g', 'b'][i], marker='^', s=80, label=f'{ch} Perturbed')
        for j in range(optimal_clusters):
            ax3.plot([base_centroids[j, 0], noisy_centroids[j, 0]], 
                     [base_centroids[j, 1], noisy_centroids[j, 1]], 
                     [base_centroids[j, 2], noisy_centroids[j, 2]], 
                     c=['r', 'g', 'b'][i], alpha=0.3)
    ax3.set_title('Centroid Shift with Perturbation')
    ax3.set_xlabel('Red')
    ax3.set_ylabel('Green')
    ax3.set_zlabel('Blue')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()

def analyze_channel_importance(image_path, client, noise_types=["gaussian"]):
    print(f"Analyzing channel importance for {image_path}")
    degradation_results = {nt: analyze_channel_degradation(image_path, client, nt) for nt in noise_types}
    dropout_results = analyze_channel_dropout(image_path, client)
    heatmap_results = {nt: generate_perturbation_heatmap(image_path, client, nt) for nt in noise_types}
    attribution_results = {nt: analyze_feature_attribution(image_path, client, nt) for nt in noise_types}
    
    channel_names = ["Red", "Green", "Blue"]
    importance_scores = {ch: 0.0 for ch in channel_names}
    for i, ch in enumerate(channel_names):
        importance_scores[ch] += dropout_results["segmentation_impact"][i]
    for nt in noise_types:
        for i, ch in enumerate(channel_names):
            importance_scores[ch] += 1 - np.mean(degradation_results[nt]["segmentation"][ch])
    
    total_score = sum(importance_scores.values())
    if total_score > 0:
        for ch in channel_names:
            importance_scores[ch] /= total_score
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(channel_names, [importance_scores[ch] for ch in channel_names], color=['r', 'g', 'b'])
    plt.title('Overall Channel Importance for Segmentation')
    plt.ylabel('Relative Importance')
    plt.ylim(0, 1)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1%}', 
                 ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
    
    return {"importance_scores": importance_scores, "degradation_results": degradation_results, 
            "dropout_results": dropout_results, "heatmap_results": heatmap_results, 
            "attribution_results": attribution_results}

# main
def main():
    client = OpenAI(api_key="API-KEY")  
    image_path = 'FILE-PATH'
    print(f"Processing image: {image_path}")
    
    stability_metrics = plot_confusion_matrices(image_path, client)
    if stability_metrics:
        print("\nStability Metrics:")
        for ch, stab in stability_metrics.items():
            print(f"{ch.capitalize()} channel stability: {stab:.3f}")
    
    results = analyze_channel_importance(image_path, client, noise_types=["gaussian"])
    print("\nChannel Importance Scores:")
    for ch, score in results["importance_scores"].items():
        print(f"{ch} channel importance: {score:.2%}")

if __name__ == "__main__":
    main()