"""
LUNA16 Dataset File Viewer
Author: Research Team
Description: Python script to explore and visualize LUNA16 lung nodule dataset files
Compatible with Spyder IDE
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import SimpleITK as sitk
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LUNA16Viewer:
    def __init__(self, base_path):
        """
        Initialize LUNA16 dataset viewer
        
        Args:
            base_path (str): Path to the desktop containing LUNA16 data
        """
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / "seg-lungs-LUNA16"
        self.annotations_file = self.base_path / "annotations.csv"
        self.candidates_file = self.base_path / "candidates.csv"
        
        print(f"Base path: {self.base_path}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Dataset exists: {self.dataset_path.exists()}")
        
    def explore_directory_structure(self):
        """Explore and display the directory structure"""
        print("\n" + "="*60)
        print("DIRECTORY STRUCTURE EXPLORATION")
        print("="*60)
        
        # List all folders in desktop
        print("\nFolders in Desktop:")
        for item in self.base_path.iterdir():
            if item.is_dir():
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                print(f"📁 {item.name}: {size / (1024*1024):.1f} MB")
        
        # List CSV files
        print("\nCSV Files in Desktop:")
        csv_files = list(self.base_path.glob("*.csv"))
        for csv_file in csv_files:
            size = csv_file.stat().st_size / 1024
            print(f"📄 {csv_file.name}: {size:.1f} KB")
        
        return csv_files
    
    def analyze_dataset_files(self):
        """Analyze the LUNA16 dataset files"""
        print("\n" + "="*60)
        print("LUNA16 DATASET FILES ANALYSIS")
        print("="*60)
        
        if not self.dataset_path.exists():
            print(f"❌ Dataset path not found: {self.dataset_path}")
            return
        
        # Get all MHD and ZRAW files
        mhd_files = list(self.dataset_path.glob("*.mhd"))
        zraw_files = list(self.dataset_path.glob("*.zraw"))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   MHD files: {len(mhd_files)}")
        print(f"   ZRAW files: {len(zraw_files)}")
        print(f"   Total files: {len(list(self.dataset_path.iterdir()))}")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in self.dataset_path.rglob('*') if f.is_file())
        print(f"   Total size: {total_size / (1024**3):.2f} GB")
        
        # Analyze file sizes
        if mhd_files or zraw_files:
            self.plot_file_size_distribution(mhd_files, zraw_files)
        
        return mhd_files, zraw_files
    
    def plot_file_size_distribution(self, mhd_files, zraw_files):
        """Plot file size distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MHD file sizes
        if mhd_files:
            mhd_sizes = [f.stat().st_size / 1024 for f in mhd_files]  # KB
            ax1.hist(mhd_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'MHD File Sizes Distribution\n({len(mhd_files)} files)')
            ax1.set_xlabel('Size (KB)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # ZRAW file sizes
        if zraw_files:
            zraw_sizes = [f.stat().st_size / (1024**2) for f in zraw_files]  # MB
            ax2.hist(zraw_sizes, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title(f'ZRAW File Sizes Distribution\n({len(zraw_files)} files)')
            ax2.set_xlabel('Size (MB)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def load_and_analyze_csv_files(self):
        """Load and analyze CSV files"""
        print("\n" + "="*60)
        print("CSV FILES ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Load annotations.csv
        if self.annotations_file.exists():
            print(f"\n📊 Loading {self.annotations_file.name}...")
            annotations = pd.read_csv(self.annotations_file)
            results['annotations'] = annotations
            
            print(f"   Shape: {annotations.shape}")
            print(f"   Columns: {list(annotations.columns)}")
            print(f"\n   First 5 rows:")
            print(annotations.head())
            
            # Plot annotations statistics
            self.plot_annotations_analysis(annotations)
        
        # Load candidates.csv
        if self.candidates_file.exists():
            print(f"\n📊 Loading {self.candidates_file.name}...")
            candidates = pd.read_csv(self.candidates_file)
            results['candidates'] = candidates
            
            print(f"   Shape: {candidates.shape}")
            print(f"   Columns: {list(candidates.columns)}")
            print(f"\n   First 5 rows:")
            print(candidates.head())
            
            # Plot candidates statistics
            self.plot_candidates_analysis(candidates)
        
        return results
    
    def plot_annotations_analysis(self, annotations):
        """Plot annotations analysis"""
        if annotations.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Diameter distribution
        if 'diameter_mm' in annotations.columns:
            axes[0, 0].hist(annotations['diameter_mm'], bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 0].set_title('Nodule Diameter Distribution')
            axes[0, 0].set_xlabel('Diameter (mm)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Coordinate distributions
        coord_cols = ['coordX', 'coordY', 'coordZ']
        colors = ['red', 'blue', 'orange']
        for i, (col, color) in enumerate(zip(coord_cols, colors)):
            if col in annotations.columns:
                axes[0, 1].hist(annotations[col], bins=20, alpha=0.5, color=color, label=col, edgecolor='black')
        axes[0, 1].set_title('Coordinate Distributions')
        axes[0, 1].set_xlabel('Coordinate Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # SeriesUID count
        if 'seriesuid' in annotations.columns:
            series_counts = annotations['seriesuid'].value_counts().head(10)
            axes[1, 0].bar(range(len(series_counts)), series_counts.values, color='purple', alpha=0.7)
            axes[1, 0].set_title('Top 10 Series by Nodule Count')
            axes[1, 0].set_xlabel('Series (index)')
            axes[1, 0].set_ylabel('Number of Nodules')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        ANNOTATIONS SUMMARY
        
        Total annotations: {len(annotations)}
        Unique series: {annotations['seriesuid'].nunique() if 'seriesuid' in annotations.columns else 'N/A'}
        
        Diameter Statistics:
        Mean: {annotations['diameter_mm'].mean():.2f} mm
        Std: {annotations['diameter_mm'].std():.2f} mm
        Min: {annotations['diameter_mm'].min():.2f} mm
        Max: {annotations['diameter_mm'].max():.2f} mm
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def plot_candidates_analysis(self, candidates):
        """Plot candidates analysis"""
        if candidates.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        if 'class' in candidates.columns:
            class_counts = candidates['class'].value_counts()
            axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                          colors=['lightcoral', 'skyblue'])
            axes[0, 0].set_title('Class Distribution\n(0: Non-nodule, 1: Nodule)')
        
        # Coordinate distributions
        coord_cols = ['coordX', 'coordY', 'coordZ']
        colors = ['red', 'blue', 'orange']
        for i, (col, color) in enumerate(zip(coord_cols, colors)):
            if col in candidates.columns:
                axes[0, 1].hist(candidates[col], bins=20, alpha=0.5, color=color, label=col, edgecolor='black')
        axes[0, 1].set_title('Coordinate Distributions')
        axes[0, 1].set_xlabel('Coordinate Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Class by coordinates scatter
        if all(col in candidates.columns for col in ['coordX', 'coordY', 'class']):
            for class_val in candidates['class'].unique():
                subset = candidates[candidates['class'] == class_val]
                axes[1, 0].scatter(subset['coordX'], subset['coordY'], 
                                 alpha=0.6, label=f'Class {class_val}', s=30)
            axes[1, 0].set_title('Candidates Distribution (X vs Y)')
            axes[1, 0].set_xlabel('X Coordinate')
            axes[1, 0].set_ylabel('Y Coordinate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        CANDIDATES SUMMARY
        
        Total candidates: {len(candidates)}
        Unique series: {candidates['seriesuid'].nunique() if 'seriesuid' in candidates.columns else 'N/A'}
        
        Class Distribution:
        Class 0 (Non-nodule): {(candidates['class'] == 0).sum() if 'class' in candidates.columns else 'N/A'}
        Class 1 (Nodule): {(candidates['class'] == 1).sum() if 'class' in candidates.columns else 'N/A'}
        
        Positive Rate: {(candidates['class'] == 1).mean()*100:.1f}%
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def load_sample_image(self, mhd_files, sample_index=0):
        """Load and display a sample medical image"""
        print("\n" + "="*60)
        print("SAMPLE IMAGE LOADING")
        print("="*60)
        
        if not mhd_files:
            print("❌ No MHD files found to load")
            return None
        
        if sample_index >= len(mhd_files):
            sample_index = 0
            
        sample_file = mhd_files[sample_index]
        print(f"\n📊 Loading sample image: {sample_file.name}")
        
        try:
            # Load using SimpleITK
            image = sitk.ReadImage(str(sample_file))
            image_array = sitk.GetArrayFromImage(image)
            
            print(f"   Image shape: {image_array.shape}")
            print(f"   Data type: {image_array.dtype}")
            print(f"   Value range: [{image_array.min():.2f}, {image_array.max():.2f}]")
            print(f"   Spacing: {image.GetSpacing()}")
            print(f"   Origin: {image.GetOrigin()}")
            
            # Display sample slices
            self.display_sample_slices(image_array, sample_file.name)
            
            return image_array
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            print("💡 Make sure SimpleITK is installed: pip install SimpleITK")
            return None
    
    def display_sample_slices(self, image_array, filename):
        """Display sample slices from the 3D image"""
        if image_array.ndim != 3:
            print(f"⚠️ Expected 3D image, got {image_array.ndim}D")
            return
        
        # Select slices to display
        z_max = image_array.shape[0]
        slice_indices = [z_max//4, z_max//2, 3*z_max//4]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, slice_idx in enumerate(slice_indices):
            if slice_idx < z_max:
                axes[i].imshow(image_array[slice_idx], cmap='gray')
                axes[i].set_title(f'Slice {slice_idx}/{z_max-1}')
                axes[i].axis('off')
        
        plt.suptitle(f'Sample Slices from: {filename}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self, csv_data, mhd_files, zraw_files):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("LUNA16 DATASET SUMMARY REPORT")
        print("="*80)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Dataset location: {self.dataset_path}")
        
        print(f"\n📁 FILE INVENTORY:")
        print(f"   MHD files: {len(mhd_files)}")
        print(f"   ZRAW files: {len(zraw_files)}")
        
        if 'annotations' in csv_data:
            ann = csv_data['annotations']
            print(f"\n📊 ANNOTATIONS DATA:")
            print(f"   Total annotations: {len(ann)}")
            print(f"   Unique series: {ann['seriesuid'].nunique()}")
            if 'diameter_mm' in ann.columns:
                print(f"   Avg nodule diameter: {ann['diameter_mm'].mean():.2f} mm")
        
        if 'candidates' in csv_data:
            cand = csv_data['candidates']
            print(f"\n🎯 CANDIDATES DATA:")
            print(f"   Total candidates: {len(cand)}")
            print(f"   Unique series: {cand['seriesuid'].nunique()}")
            if 'class' in cand.columns:
                pos_rate = (cand['class'] == 1).mean() * 100
                print(f"   Positive rate: {pos_rate:.1f}%")
        
        print(f"\n✅ Dataset ready for analysis!")
        print("="*80)


def main():
    """Main function to run the LUNA16 dataset viewer"""
    # Set your desktop path
    desktop_path = r"C:\Users\Usman Traders\Desktop"
    
    # Initialize viewer
    viewer = LUNA16Viewer(desktop_path)
    
    # 1. Explore directory structure
    csv_files = viewer.explore_directory_structure()
    
    # 2. Analyze dataset files
    mhd_files, zraw_files = viewer.analyze_dataset_files()
    
    # 3. Load and analyze CSV files
    csv_data = viewer.load_and_analyze_csv_files()
    
    # 4. Load sample image (optional - requires SimpleITK)
    if mhd_files:
        sample_image = viewer.load_sample_image(mhd_files, sample_index=0)
    
    # 5. Generate summary report
    viewer.generate_summary_report(csv_data, mhd_files, zraw_files)
    
    return viewer, csv_data, mhd_files, zraw_files


# Execute if running directly
if __name__ == "__main__":
    print("🔬 LUNA16 Dataset Viewer Starting...")
    print("📍 Make sure your file paths are correct!")
    
    # Run main analysis
    viewer, csv_data, mhd_files, zraw_files = main()
    
    print("\n🎉 Analysis complete! Variables available:")
    print("   - viewer: LUNA16Viewer object")
    print("   - csv_data: Dictionary with loaded CSV data")
    print("   - mhd_files: List of MHD files")
    print("   - zraw_files: List of ZRAW files")