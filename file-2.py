import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from pathlib import Path
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LIDCDatasetAnalyzer:
    def __init__(self, base_path):
        """
        Initialize the LIDC dataset analyzer
        
        Args:
            base_path (str): Path to LIDC-IDRI-slices folder
        """
        self.base_path = Path(base_path)
        self.dataset_info = defaultdict(list)
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        """Load and analyze the dataset structure"""
        print("🔍 Analyzing dataset structure...")
        
        # Get all patient folders
        patient_folders = [f for f in self.base_path.iterdir() if f.is_dir() and f.name.startswith('LIDC-IDRI-')]
        
        total_nodules = 0
        total_images = 0
        total_masks = 0
        
        for patient_folder in patient_folders:
            patient_id = patient_folder.name
            
            # Get nodule folders for this patient
            nodule_folders = [f for f in patient_folder.iterdir() if f.is_dir() and f.name.startswith('nodule-')]
            
            for nodule_folder in nodule_folders:
                nodule_id = nodule_folder.name
                
                # Count images and masks
                images_path = nodule_folder / 'images'
                masks_paths = [nodule_folder / f'mask-{i}' for i in range(10)]  # Assuming up to 10 masks
                
                if images_path.exists():
                    image_files = list(images_path.glob('*.png'))
                    num_images = len(image_files)
                    total_images += num_images
                    
                    # Count masks
                    num_masks = 0
                    for mask_path in masks_paths:
                        if mask_path.exists():
                            mask_files = list(mask_path.glob('*.png'))
                            num_masks += len(mask_files)
                    
                    total_masks += num_masks
                    total_nodules += 1
                    
                    # Store information
                    self.dataset_info['patient_id'].append(patient_id)
                    self.dataset_info['nodule_id'].append(nodule_id)
                    self.dataset_info['num_images'].append(num_images)
                    self.dataset_info['num_masks'].append(num_masks)
                    self.dataset_info['masks_per_image'].append(num_masks / num_images if num_images > 0 else 0)
        
        # Create DataFrame
        self.df = pd.DataFrame(self.dataset_info)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total patients: {len(patient_folders)}")
        print(f"🎯 Total nodules: {total_nodules}")
        print(f"🖼️ Total images: {total_images}")
        print(f"🎭 Total masks: {total_masks}")
    
    def display_dataset_summary(self):
        """Display comprehensive dataset summary"""
        print("\n" + "="*60)
        print("📋 LIDC-IDRI DATASET SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"Total Patients: {self.df['patient_id'].nunique()}")
        print(f"Total Nodules: {len(self.df)}")
        print(f"Average Nodules per Patient: {len(self.df) / self.df['patient_id'].nunique():.2f}")
        print(f"Total Images: {self.df['num_images'].sum()}")
        print(f"Total Masks: {self.df['num_masks'].sum()}")
        
        # Images per nodule statistics
        print(f"\nImages per Nodule:")
        print(f"  Mean: {self.df['num_images'].mean():.2f}")
        print(f"  Median: {self.df['num_images'].median():.2f}")
        print(f"  Min: {self.df['num_images'].min()}")
        print(f"  Max: {self.df['num_images'].max()}")
        
        # Masks per nodule statistics
        print(f"\nMasks per Nodule:")
        print(f"  Mean: {self.df['num_masks'].mean():.2f}")
        print(f"  Median: {self.df['num_masks'].median():.2f}")
        print(f"  Min: {self.df['num_masks'].min()}")
        print(f"  Max: {self.df['num_masks'].max()}")
    
    def plot_distribution_analysis(self):
        """Create comprehensive distribution plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi = 300)
        fig.suptitle('LIDC-IDRI Dataset Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Nodules per patient
        nodules_per_patient = self.df.groupby('patient_id').size()
        axes[0, 0].hist(nodules_per_patient.values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Distribution of Nodules per Patient')
        axes[0, 0].set_xlabel('Number of Nodules')
        axes[0, 0].set_ylabel('Number of Patients')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Images per nodule
        axes[0, 1].hist(self.df['num_images'], bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Distribution of Images per Nodule')
        axes[0, 1].set_xlabel('Number of Images')
        axes[0, 1].set_ylabel('Number of Nodules')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Masks per nodule
        axes[0, 2].hist(self.df['num_masks'], bins=30, alpha=0.7, edgecolor='black', color='green')
        axes[0, 2].set_title('Distribution of Masks per Nodule')
        axes[0, 2].set_xlabel('Number of Masks')
        axes[0, 2].set_ylabel('Number of Nodules')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot - Images per nodule
        axes[1, 0].boxplot(self.df['num_images'], patch_artist=True, 
                          boxprops=dict(facecolor='lightblue'))
        axes[1, 0].set_title('Images per Nodule (Box Plot)')
        axes[1, 0].set_ylabel('Number of Images')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Box plot - Masks per nodule
        axes[1, 1].boxplot(self.df['num_masks'], patch_artist=True,
                          boxprops=dict(facecolor='lightgreen'))
        axes[1, 1].set_title('Masks per Nodule (Box Plot)')
        axes[1, 1].set_ylabel('Number of Masks')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Masks per image ratio
        axes[1, 2].hist(self.df['masks_per_image'], bins=20, alpha=0.7, edgecolor='black', color='purple')
        axes[1, 2].set_title('Distribution of Masks per Image Ratio')
        axes[1, 2].set_xlabel('Masks per Image Ratio')
        axes[1, 2].set_ylabel('Number of Nodules')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_sample_data(self, num_samples=6):
        """Visualize sample images and their corresponding masks"""
        print(f"\n🖼️ Visualizing {num_samples} random samples...")
        
        # Select random samples
        sample_indices = np.random.choice(len(self.df), min(num_samples, len(self.df)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples), dpi = 300)
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Sample LIDC-IDRI Images and Masks', fontsize=16, fontweight='bold')
        
        for idx, sample_idx in enumerate(sample_indices):
            row = self.df.iloc[sample_idx]
            patient_id = row['patient_id']
            nodule_id = row['nodule_id']
            
            # Get paths
            images_path = self.base_path / patient_id / nodule_id / 'images'
            
            if images_path.exists():
                # Get a random image from this nodule
                image_files = list(images_path.glob('*.png'))
                if image_files:
                    # Load and display image
                    img_path = np.random.choice(image_files)
                    image = Image.open(img_path)
                    image_array = np.array(image)
                    
                    axes[idx, 0].imshow(image_array, cmap='gray')
                    axes[idx, 0].set_title(f'{patient_id}\n{nodule_id}\nOriginal Image')
                    axes[idx, 0].axis('off')
                    
                    # Try to load corresponding masks
                    mask_loaded = False
                    for mask_num in range(4):  # Try first 4 masks
                        mask_path = self.base_path / patient_id / nodule_id / f'mask-{mask_num}'
                        if mask_path.exists():
                            mask_files = list(mask_path.glob('*.png'))
                            if mask_files:
                                # Find corresponding mask file
                                img_name = img_path.stem
                                mask_file = mask_path / f'{img_name}.png'
                                if mask_file.exists():
                                    mask = Image.open(mask_file)
                                    mask_array = np.array(mask)
                                    
                                    if mask_num < 3:  # Show first 3 masks
                                        axes[idx, mask_num + 1].imshow(mask_array, cmap='jet', alpha=0.8)
                                        axes[idx, mask_num + 1].set_title(f'Mask {mask_num}')
                                        axes[idx, mask_num + 1].axis('off')
                                        mask_loaded = True
                    
                    # Fill empty mask slots
                    for col in range(1, 4):
                        if not mask_loaded or col > 3:
                            axes[idx, col].text(0.5, 0.5, 'No Mask\nAvailable', 
                                              ha='center', va='center', transform=axes[idx, col].transAxes,
                                              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                            axes[idx, col].set_xlim(0, 1)
                            axes[idx, col].set_ylim(0, 1)
                            axes[idx, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_properties(self, sample_size=100):
        """Analyze image properties like dimensions, intensity distributions"""
        print(f"\n🔍 Analyzing image properties from {sample_size} random samples...")
        
        image_info = {
            'width': [],
            'height': [],
            'mean_intensity': [],
            'std_intensity': [],
            'min_intensity': [],
            'max_intensity': []
        }
        
        # Sample random images
        samples_collected = 0
        sample_indices = np.random.choice(len(self.df), min(sample_size, len(self.df)), replace=False)
        
        for sample_idx in sample_indices:
            if samples_collected >= sample_size:
                break
                
            row = self.df.iloc[sample_idx]
            patient_id = row['patient_id']
            nodule_id = row['nodule_id']
            
            images_path = self.base_path / patient_id / nodule_id / 'images'
            
            if images_path.exists():
                image_files = list(images_path.glob('*.png'))
                if image_files:
                    # Analyze first image in the nodule
                    img = Image.open(image_files[0])
                    img_array = np.array(img)
                    
                    image_info['width'].append(img_array.shape[1])
                    image_info['height'].append(img_array.shape[0])
                    image_info['mean_intensity'].append(np.mean(img_array))
                    image_info['std_intensity'].append(np.std(img_array))
                    image_info['min_intensity'].append(np.min(img_array))
                    image_info['max_intensity'].append(np.max(img_array))
                    
                    samples_collected += 1
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi = 300)
        fig.suptitle('Image Properties Analysis', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(image_info['width'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(image_info['height'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mean intensity distribution
        axes[0, 2].hist(image_info['mean_intensity'], bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[0, 2].set_title('Mean Intensity Distribution')
        axes[0, 2].set_xlabel('Mean Intensity')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Standard deviation distribution
        axes[1, 0].hist(image_info['std_intensity'], bins=20, alpha=0.7, edgecolor='black', color='red')
        axes[1, 0].set_title('Intensity Standard Deviation')
        axes[1, 0].set_xlabel('Standard Deviation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Min vs Max intensity scatter
        axes[1, 1].scatter(image_info['min_intensity'], image_info['max_intensity'], alpha=0.6)
        axes[1, 1].set_title('Min vs Max Intensity')
        axes[1, 1].set_xlabel('Min Intensity')
        axes[1, 1].set_ylabel('Max Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Width vs Height scatter
        axes[1, 2].scatter(image_info['width'], image_info['height'], alpha=0.6, color='purple')
        axes[1, 2].set_title('Width vs Height')
        axes[1, 2].set_xlabel('Width (pixels)')
        axes[1, 2].set_ylabel('Height (pixels)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n📊 Image Properties Statistics:")
        for prop, values in image_info.items():
            if values:
                print(f"{prop.replace('_', ' ').title()}: "
                      f"Mean={np.mean(values):.2f}, "
                      f"Std={np.std(values):.2f}, "
                      f"Min={np.min(values):.2f}, "
                      f"Max={np.max(values):.2f}")
    
    def create_comprehensive_report(self):
        """Generate a comprehensive EDA report"""
        print("\n" + "="*80)
        print("🏥 COMPREHENSIVE LIDC-IDRI DATASET ANALYSIS REPORT")
        print("="*80)
        
        # Display summary
        self.display_dataset_summary()
        
        # Create all visualizations
        self.plot_distribution_analysis()
        self.visualize_sample_data(num_samples=6)
        self.analyze_image_properties(sample_size=100)
        
        # Additional insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 40)
        
        # Dataset balance
        avg_nodules_per_patient = len(self.df) / self.df['patient_id'].nunique()
        print(f"• Average nodules per patient: {avg_nodules_per_patient:.2f}")
        
        # Image consistency
        images_per_nodule_std = self.df['num_images'].std()
        print(f"• Images per nodule variability (std): {images_per_nodule_std:.2f}")
        
        # Annotation coverage
        total_images = self.df['num_images'].sum()
        total_masks = self.df['num_masks'].sum()
        annotation_ratio = total_masks / total_images if total_images > 0 else 0
        print(f"• Average annotation coverage: {annotation_ratio:.2f} masks per image")
        
        # Data sparsity
        nodules_with_few_images = (self.df['num_images'] < 5).sum()
        print(f"• Nodules with <5 images: {nodules_with_few_images} ({nodules_with_few_images/len(self.df)*100:.1f}%)")
        
        print("\n✅ Analysis complete! Your LIDC-IDRI dataset is ready for further processing.")

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your dataset path
    dataset_path = r"C:\Users\Usman Traders\Desktop\LIDC-IDRI-slices"
    
    # Create analyzer instance
    analyzer = LIDCDatasetAnalyzer(dataset_path)
    
    # Run comprehensive analysis
    analyzer.create_comprehensive_report()
    
    # You can also run individual analyses:
    # analyzer.display_dataset_summary()
    # analyzer.plot_distribution_analysis()
    # analyzer.visualize_sample_data(num_samples=8)
    # analyzer.analyze_image_properties(sample_size=150)