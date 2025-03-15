import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
from openai import OpenAI
import warnings
import os
warnings.filterwarnings('ignore')

class MoS2Analyzer:
   
    FEATURE_LABELS = {
        0: "Single Layer",
        1: "Multi Layer",
        2: "Substrate",
        3: "Defects",
        4: "Edge Sites"
    }
    
    def __init__(self, openai_api_key):
        """Initialize the analyzer with necessary models."""
        self.client = OpenAI(api_key=openai_api_key)
        
        print("Loading SAM model...")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        if torch.cuda.is_available():
            self.sam_model.to("cuda")
        print("Models loaded successfully!")

    def get_sam_prompt(self, image_description):
        """Get improved SAM prompting strategy with better defect detection."""
        prompt = f"""
        Given an image of MoS2 crystal with the following description:
        {image_description}
        
        Provide specific prompting points (x,y coordinates) focusing especially on defects, which appear as bright blue spots.
        Key features to identify:
        1. Single layer regions (larger uniform areas)
        2. Multi-layer regions (darker regions)
        3. Substrate (background)
        4. Defects (bright blue spots and irregularities)
        5. Edge sites (boundaries between regions)
        
        Pay special attention to:
        - Bright blue spots which indicate defects
        - Contrast differences that separate layers
        - Clear boundaries between different regions
        
        Response format should be a Python dictionary with relative coordinates (0-1 range) like:
        {{"single_layer": [(0.3, 0.4), (0.5, 0.6)],
         "multi_layer": [(0.7, 0.8)],
         "substrate": [(0.2, 0.2)],
         "defects": [(0.4, 0.7), (0.6, 0.3), (0.5, 0.5)],  # Include more defect points
         "edge_sites": [(0.6, 0.3)]}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in 2D materials analysis, particularly skilled at identifying defects in MoS2 samples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            try:
                return eval(response.choices[0].message.content)
            except:
                # Enhanced default points with focus on defects
                return {
                    "single_layer": [(0.3, 0.4), (0.5, 0.6)],
                    "multi_layer": [(0.7, 0.8), (0.2, 0.7)],
                    "substrate": [(0.2, 0.2), (0.8, 0.8)],
                    "defects": [(0.4, 0.7), (0.6, 0.3), (0.5, 0.5), (0.3, 0.3)],
                    "edge_sites": [(0.6, 0.3), (0.4, 0.4)]
                }
        except Exception as e:
            print(f"Error getting SAM prompts: {e}")
            return None

    def load_and_verify_image(self, image_path):
        """Load and verify image with detailed error handling."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def process_image(self, image_path, prompt_points=None):
        """Process image with SAM using provided or generated prompt points."""
        try:
            print(f"Loading image from: {image_path}")
            image = self.load_and_verify_image(image_path)
            print("Image loaded successfully!")
            
            if prompt_points is None:
                image_desc = "MoS2 crystal sample with visible layer contrasts, bright blue defect spots, and edge features"
                prompt_points = self.get_sam_prompt(image_desc)
            
            h, w = image.shape[:2]
            pixel_points = []
            labels = []

            for feature, points in prompt_points.items():
                for x, y in points:
                    pixel_points.append([int(x * w), int(y * h)])
                    labels.append(list(prompt_points.keys()).index(feature))

            inputs = self.sam_processor(image, input_points=[pixel_points], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = self.sam_model(**inputs)
            masks = outputs.pred_masks.squeeze().cpu().detach().numpy()
            
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=0)
            
            return image, masks, labels
            
        except Exception as e:
            print(f"Error in process_image: {str(e)}")
            raise

    def apply_rgb_perturbation(self, image, channel, magnitude=50):
        """Apply random perturbation to specified RGB channel."""
        perturbed = image.copy()
        noise = np.random.normal(0, magnitude, image.shape[:2])
        perturbed[:, :, channel] = np.clip(perturbed[:, :, channel] + noise, 0, 255)
        return perturbed

    def analyze_perturbations(self, image_path):
        """Perform complete perturbation analysis with enhanced defect detection."""
        try:
            print("Starting perturbation analysis...")
            image, base_masks, feature_labels = self.process_image(image_path)
            print("Base segmentation completed.")
            
            results = []
            for channel in range(3):  # Process all RGB channels
                print(f"Processing {'RGB'[channel]} channel perturbation...")
                perturbed_image = self.apply_rgb_perturbation(image, channel)
                _, perturbed_masks, _ = self.process_image(image_path)
                
                if len(perturbed_masks.shape) == 2:
                    perturbed_masks = np.expand_dims(perturbed_masks, axis=0)
                
                results.append((f"{'RGB'[channel]}", perturbed_masks))
            
            print("Perturbation analysis completed.")
            return image, base_masks, results, feature_labels
        
        except Exception as e:
            print(f"Error in analyze_perturbations: {str(e)}")
            raise

    def plot_segmentation(self, image, base_masks, perturbed_results):
        """Plot original image and segmentation results with uniform size."""
        try:
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(image)
            ax1.set_title('Original MoS2 Image', fontsize=14, pad=20)
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[0, 1])
            base_mask_vis = np.max(base_masks, axis=0) if len(base_masks.shape) == 3 else base_masks
            ax2.imshow(base_mask_vis, cmap='nipy_spectral')
            ax2.set_title('Base Segmentation', fontsize=14, pad=20)
            ax2.axis('off')
            

            for idx, (channel, perturbed_masks) in enumerate(perturbed_results):
                if idx >= 2: 
                    continue
                ax = fig.add_subplot(gs[1, idx])
                perturbed_mask_vis = np.max(perturbed_masks, axis=0) if len(perturbed_masks.shape) == 3 else perturbed_masks
                ax.imshow(perturbed_mask_vis, cmap='nipy_spectral')
                ax.set_title(f'{channel} Channel Perturbed', fontsize=14, pad=20)
                ax.axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_segmentation: {str(e)}")
            return None

    def plot_confusion_matrices(self, base_masks, perturbed_results):
        """Plot square confusion matrices with uniform size."""
        try:
            fig = plt.figure(figsize=(24, 8))
            
            max_labels = 0
            for _, perturbed_masks in perturbed_results:
                base_labels = np.argmax(base_masks, axis=0) if len(base_masks.shape) == 3 else base_masks
                perturbed_labels = np.argmax(perturbed_masks, axis=0) if len(perturbed_masks.shape) == 3 else perturbed_masks
                n_labels = len(np.unique(np.concatenate([base_labels.flatten(), perturbed_labels.flatten()])))
                max_labels = max(max_labels, n_labels)
            
            for idx, (channel, perturbed_masks) in enumerate(perturbed_results):
                ax = fig.add_subplot(1, 3, idx + 1)
                
                base_labels = np.argmax(base_masks, axis=0) if len(base_masks.shape) == 3 else base_masks
                perturbed_labels = np.argmax(perturbed_masks, axis=0) if len(perturbed_masks.shape) == 3 else perturbed_masks
                
                cm = confusion_matrix(base_labels.flatten(), 
                                    perturbed_labels.flatten(),
                                    labels=range(max_labels))
                cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
                
                label_names = [self.FEATURE_LABELS.get(i, f"Class {i}") for i in range(max_labels)]
                
                sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=label_names,
                           yticklabels=label_names,
                           ax=ax,
                           square=True)
                
                ax.set_title(f'{channel} Channel Stability', fontsize=14, pad=20)
                ax.set_xlabel('Perturbed Features', fontsize=12)
                ax.set_ylabel('Original Features', fontsize=12)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                plt.setp(ax.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_confusion_matrices: {str(e)}")
            return None

def main():
    try:
        analyzer = MoS2Analyzer(openai_api_key="API-KEY")
        
        image_path = "file-path"

        print(f"Processing image at: {image_path}")
        
        image, base_masks, perturbed_results, feature_labels = analyzer.analyze_perturbations(image_path)
        
        fig_seg = analyzer.plot_segmentation(image, base_masks, perturbed_results)
        if fig_seg:
            plt.show()
        
        fig_conf = analyzer.plot_confusion_matrices(base_masks, perturbed_results)
        if fig_conf:
            plt.show()
        
        print("\nFeature Stability Analysis:")
        for channel, masks in perturbed_results:
            base_labels = np.argmax(base_masks, axis=0) if len(base_masks.shape) == 3 else base_masks
            perturbed_labels = np.argmax(masks, axis=0) if len(masks.shape) == 3 else masks
            
            unique_labels = np.unique(np.concatenate([base_labels.flatten(), perturbed_labels.flatten()]))
            n_labels = len(unique_labels)
            
            cm = confusion_matrix(base_labels.flatten(), 
                                perturbed_labels.flatten(),
                                labels=range(n_labels))
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            print(f"\n{channel} Channel Stability:")
            for i in range(n_labels):
                feature_name = analyzer.FEATURE_LABELS.get(i, f"Class {i}")
                print(f"{feature_name}: {cm_norm[i,i]:.3f}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please ensure you have:")
        print("1. Specified the correct image path")
        print("2. Set your OpenAI API key")
        print("3. Installed all required packages")

if __name__ == "__main__":
    main()