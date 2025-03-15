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
        self.client = OpenAI(api_key=openai_api_key)
        print("Loading SAM model...")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        if torch.cuda.is_available():
            self.sam_model.to("cuda")
        print("Models loaded successfully!")

    def default_prompts(self):
        return {
            "single_layer": [(0.3, 0.4), (0.5, 0.6)],
            "multi_layer": [(0.7, 0.8), (0.2, 0.7)],
            "substrate": [(0.2, 0.2), (0.8, 0.8)],
            "defects": [(0.4, 0.7), (0.6, 0.3)],
            "edge_sites": [(0.6, 0.3), (0.4, 0.4)]
        }

    def process_image(self, image_path=None, image_array=None, prompt_points=None):
        try:
            if image_array is not None:
                image = image_array.astype(np.uint8)
            else:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            all_masks = np.zeros((len(self.FEATURE_LABELS), h, w), dtype=np.uint8)
            
            prompt_points = prompt_points or self.default_prompts()

            for feature_idx, feature_name in self.FEATURE_LABELS.items():
                feature_key = feature_name.lower().replace(" ", "_")
                if feature_key not in prompt_points:
                    continue

                points = [(int(x*w), int(y*h)) for x, y in prompt_points[feature_key]]
                if not points:
                    continue

                inputs = self.sam_processor(
                    image,
                    input_points=[points],
                    input_labels=[[1]*len(points)],
                    return_tensors="pt"
                ).to(self.sam_model.device)

                with torch.no_grad():
                    outputs = self.sam_model(**inputs)

                masks = self.sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].tolist(),
                    inputs["reshaped_input_sizes"].tolist()
                )[0]

                if len(masks.shape) == 3 and masks.shape[0] > 0:
                    best_mask = masks[outputs.iou_scores.argmax().item()].numpy()
                    all_masks[feature_idx] = best_mask.astype(np.uint8)

            return image, all_masks
        except Exception as e:
            print(f"Error processing image: {e}")
            raise

    def apply_rgb_perturbation(self, image, channel, magnitude=50):
        try:
            perturbed = image.copy().astype(np.float32)
            noise = np.random.normal(0, magnitude, image.shape[:2])
            perturbed[:, :, channel] = np.clip(perturbed[:, :, channel] + noise, 0, 255)
            return perturbed.astype(np.uint8)
        except Exception as e:
            print(f"Error in RGB perturbation: {e}")
            raise

    def analyze_perturbations(self, image_path):
        try:
            print("Starting analysis...")
            image, base_masks = self.process_image(image_path)
            results = []
            
            for channel in range(3):  # RGB channels
                print(f"Perturbing {['R','G','B'][channel]} channel...")
                perturbed = self.apply_rgb_perturbation(image, channel)
                _, perturbed_masks = self.process_image(image_array=perturbed)
                results.append((f"{['R','G','B'][channel]}", perturbed_masks))
            
            return image, base_masks, results
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            raise

    def plot_segmentation(self, image, base_masks, perturbed_results):
        try:
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.2)
            
            # Original image
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(image)
            ax1.set_title('Original MoS2 Image', fontsize=14, pad=20)
            ax1.axis('off')
            
            # Base segmentation
            ax2 = fig.add_subplot(gs[0, 1])
            base_seg = np.argmax(base_masks, axis=0)
            ax2.imshow(base_seg, cmap='nipy_spectral')
            ax2.set_title('Base Segmentation', fontsize=14, pad=20)
            ax2.axis('off')
            
            # Perturbation results
            for idx, (channel, perturbed_masks) in enumerate(perturbed_results[:2]):
                ax = fig.add_subplot(gs[1, idx])
                perturbed_seg = np.argmax(perturbed_masks, axis=0)
                ax.imshow(perturbed_seg, cmap='nipy_spectral')
                ax.set_title(f'{channel} Channel Perturbed', fontsize=14, pad=20)
                ax.axis('off')
            
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error in plot_segmentation: {str(e)}")
            return None

    def plot_confusion_matrices(self, base_masks, perturbed_results):
        try:
            fig = plt.figure(figsize=(24, 8))
            plt.subplots_adjust(wspace=0.3)
            n_classes = len(self.FEATURE_LABELS)
            
            for idx, (channel, perturbed_masks) in enumerate(perturbed_results):
                ax = fig.add_subplot(1, 3, idx + 1)
                
                base_labels = np.argmax(base_masks, axis=0).flatten()
                perturbed_labels = np.argmax(perturbed_masks, axis=0).flatten()
                
                cm = confusion_matrix(
                    base_labels,
                    perturbed_labels,
                    labels=range(n_classes)
                )
                
                row_sums = cm.sum(axis=1)
                cm_norm = np.zeros_like(cm, dtype=float)
                for i in range(len(row_sums)):
                    if row_sums[i] != 0:
                        cm_norm[i] = cm[i] / row_sums[i]
                
                sns.heatmap(
                    cm_norm,
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    xticklabels=[self.FEATURE_LABELS[i] for i in range(n_classes)],
                    yticklabels=[self.FEATURE_LABELS[i] for i in range(n_classes)],
                    ax=ax,
                    square=True,
                    cbar_kws={'shrink': .8}
                )
                
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
        image, base_masks, results = analyzer.analyze_perturbations("FILE-PATH")
        
        analyzer.plot_segmentation(image, base_masks, results)
        analyzer.plot_confusion_matrices(base_masks, results)
        plt.show()

        print("\nStability Metrics:")
        base_seg = np.argmax(base_masks, axis=0)
        for channel, perturbed_masks in results:
            perturbed_seg = np.argmax(perturbed_masks, axis=0)
            accuracy = np.mean(base_seg == perturbed_seg)
            print(f"{channel}: {accuracy:.3f}")

    except Exception as e:
        print(f"Main execution failed: {str(e)}")

if __name__ == "__main__":
    main()