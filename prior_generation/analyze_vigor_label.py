import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any
import multiprocessing
from functools import partial
import json
from datetime import datetime
from scipy import stats
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

'''
This file is used to compile category statistics of the BEV layout maps in the VIGORv2 dataset.

'''




# Set matplotlib global style
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.transparent': True,
    'savefig.facecolor': 'none',
    'savefig.edgecolor': 'none',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14
})

# Set seaborn style
sns.set_style("whitegrid", {
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'grid.color': 'gray',
    'grid.alpha': 0.3,
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14
})

# Set seaborn font
sns.set(font='Times New Roman')
sns.set_style("whitegrid", {"font.family": "Times New Roman"})

# Define color to category mapping
COLOR_MAP = {
    (255, 178, 102): 'Building',
    (64, 90, 255): 'Parking',
    (102, 255, 102): 'GrassPark',  # Simplified category name for Grass/Park/Playground
    (0, 153, 0): 'Forest',
    (204, 229, 255): 'Water',
    (192, 192, 192): 'Path',
    (96, 96, 96): 'Road',
    (255, 255, 255): 'Background'
}

# Define statistical plot color mapping
PLOT_COLORS = {
    'Building': '#C8B4AA', # Soft beige/light taupe
    'Parking': '#A0A0A0', # Neutral gray
    'GrassPark': '#8CBE82', # Soft grass green
    'Forest': '#648C5A', # Darker olive or forest green
    'Water': '#82AAC8', # More natural light blue or blue-green
    'Path': '#D2C8B4', # Light earth yellow or beige
    'Road': '#646464', # Dark gray
    'Background': '#DCDCDC' # Light gray background
}

class LabelAnalyzer:
    def __init__(self, data_folder: str, num_workers: int = None):
        self.data_folder = data_folder
        self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        self.distributions = []
        self.city_stats = defaultdict(list)
        self.num_workers = num_workers or multiprocessing.cpu_count()
        print(f"Using {self.num_workers} workers for parallel processing")
        
    def analyze_single_image(self, label_path: str) -> Dict[str, float]:
        """Analyze a single label image"""
        try:
            label = cv2.imread(label_path)
            if label is None:
                print(f"Warning: Could not read image {label_path}")
                return None
                
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            total_pixels = label.shape[0] * label.shape[1]
            class_distribution = defaultdict(float)
            
            # Merge Grass/Park/Playground pixels
            grass_park_mask = np.zeros(label.shape[:2], dtype=bool)
            for color, class_name in COLOR_MAP.items():
                if class_name == 'GrassPark':
                    mask = np.all(label == color, axis=-1)
                    grass_park_mask = np.logical_or(grass_park_mask, mask)
                else:
                    mask = np.all(label == color, axis=-1)
                    class_distribution[class_name] = np.sum(mask) / total_pixels
            
            # Add merged GrassPark category
            class_distribution['GrassPark'] = np.sum(grass_park_mask) / total_pixels
                
            return dict(class_distribution)
        except Exception as e:
            print(f"Error processing {label_path}: {str(e)}")
            return None
    
    def process_batch(self, label_files: List[str], map_folder: str) -> List[Dict[str, float]]:
        """Process a batch of image files"""
        results = []
        for label_file in label_files:
            label_path = os.path.join(map_folder, label_file)
            result = self.analyze_single_image(label_path)
            if result:
                results.append(result)
        return results
    
    def analyze_city(self, city: str) -> List[Dict[str, float]]:
        """Analyze all labels for a single city"""
        map_folder = os.path.join(self.data_folder, city, 'map')
        if not os.path.exists(map_folder):
            print(f"Warning: {map_folder} does not exist")
            return []
            
        print(f"\nAnalyzing {city}...")
        label_files = [f for f in os.listdir(map_folder) if f.endswith('.png')]
        
        # Split file list into multiple batches
        batch_size = max(1, len(label_files) // self.num_workers)
        batches = [label_files[i:i + batch_size] for i in range(0, len(label_files), batch_size)]
        
        city_distributions = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            process_batch_partial = partial(self.process_batch, map_folder=map_folder)
            
            futures = list(tqdm(
                executor.map(process_batch_partial, batches),
                total=len(batches),
                desc=f"Processing {city}"
            ))
            
            for batch_results in futures:
                city_distributions.extend(batch_results)
                self.city_stats[city].extend(batch_results)
                
        return city_distributions
    
    def analyze_all_cities(self):
        """Analyze label distributions for all cities"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cities)) as executor:
            city_futures = list(tqdm(
                executor.map(self.analyze_city, self.cities),
                total=len(self.cities),
                desc="Processing cities"
            ))
            
            for city_distributions in city_futures:
                self.distributions.extend(city_distributions)
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate statistical information"""
        df = pd.DataFrame(self.distributions)
        stats = {
            'overall': df.describe(),
            'by_city': {}
        }
        
        for city, distributions in self.city_stats.items():
            city_df = pd.DataFrame(distributions)
            stats['by_city'][city] = city_df.describe()
            
        return stats
    
    def plot_distributions(self, output_dir: str):
        """Plot and save multiple distribution plots"""
        df = pd.DataFrame(self.distributions)
        
        # Create statistics directory structure
        stats_dir = os.path.join(output_dir, 'statistics')
        os.makedirs(stats_dir, exist_ok=True)
        
        # Create subdirectories for each city
        for city in self.cities:
            city_dir = os.path.join(stats_dir, city)
            os.makedirs(city_dir, exist_ok=True)
            
        # Create class summary directory
        class_summary_dir = os.path.join(stats_dir, 'class_summary')
        os.makedirs(class_summary_dir, exist_ok=True)
        
        # Set seaborn palette
        sns.set_palette([PLOT_COLORS[col] for col in df.columns])
        
        # 1. Overall distribution plot (with density curves)
        plt.figure(figsize=(15, 10))
        for column in df.columns:
            sns.kdeplot(data=df[column], label=column, fill=True, alpha=0.3, color=PLOT_COLORS[column])
        plt.title('Overall Class Distribution with Density Curves', fontsize=14)
        plt.xlabel('Pixel Ratio', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, 'overall_distribution.png'), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        # 2. City comparison heatmap
        plt.figure(figsize=(12, 8))
        city_means = pd.DataFrame({city: pd.DataFrame(dists).mean() 
                                 for city, dists in self.city_stats.items()})
        sns.heatmap(city_means, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Mean Class Distribution by City', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, 'city_comparison_heatmap.png'), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        # 3. Class correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Class Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight', transparent=True)
        plt.close()
        
        # 4. Class distribution for each city
        for city, dists in self.city_stats.items():
            city_df = pd.DataFrame(dists)
            city_dir = os.path.join(stats_dir, city)
            
            # 4.1 Class distribution histograms for all classes in the city
            plt.figure(figsize=(15, 10))
            for column in city_df.columns:
                sns.histplot(data=city_df[column], 
                           kde=True, 
                           stat='density', 
                           color=PLOT_COLORS[column],
                           label=column,
                           alpha=0.5)
            plt.title(f'{city} - Class Distribution Histograms', fontsize=14)
            plt.xlabel('Pixel Ratio', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(city_dir, 'class_distributions.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 4.2 Individual distribution plot for each class in the city
            for column in city_df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=city_df[column], 
                           kde=True, 
                           stat='density', 
                           color=PLOT_COLORS[column])
                plt.title(f'{city} - {column} Distribution', fontsize=14)
                plt.xlabel('Pixel Ratio', fontsize=12)
                plt.ylabel('Density', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(city_dir, f'{column}_distribution.png'), dpi=300, bbox_inches='tight', transparent=True)
                plt.close()
            
            # 4.3 Class distribution violin plot for the city
            plt.figure(figsize=(12, 8))
            violin_parts = plt.violinplot([city_df[col] for col in city_df.columns], showmeans=True)
            for i, pc in enumerate(violin_parts['bodies']):
                pc.set_facecolor(PLOT_COLORS[city_df.columns[i]])
                pc.set_alpha(0.7)
            plt.xticks(range(1, len(city_df.columns) + 1), city_df.columns, rotation=45, ha='right')
            plt.title(f'{city} - Class Distribution Violin Plot', fontsize=14)
            plt.ylabel('Pixel Ratio', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(city_dir, 'class_violin_plot.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 4.4 Class correlation matrix for the city
            plt.figure(figsize=(10, 8))
            city_corr = city_df.corr()
            sns.heatmap(city_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title(f'{city} - Class Correlation Matrix', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(city_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
        
        # 5. City comparison for each class
        for column in df.columns:
            plt.figure(figsize=(12, 8))
            city_data = []
            for city, dists in self.city_stats.items():
                city_df = pd.DataFrame(dists)
                city_data.append(city_df[column])
            
            violin_parts = plt.violinplot(city_data, showmeans=True)
            for pc in violin_parts['bodies']:
                pc.set_facecolor(PLOT_COLORS[column])
                pc.set_alpha(0.7)
            
            plt.xticks(range(1, len(self.cities) + 1), self.cities, rotation=45, ha='right')
            plt.title(f'{column} Distribution by City', fontsize=14)
            plt.ylabel('Pixel Ratio', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, f'{column}_city_comparison.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
        # 6. Summary analysis for each class
        for column in df.columns:
            # 6.1 Create class summary directory
            class_dir = os.path.join(class_summary_dir, column)
            os.makedirs(class_dir, exist_ok=True)
            
            # 6.2 Class histogram and density curve
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df[column], kde=True, stat='density', color=PLOT_COLORS[column])
            plt.title(f'{column} Distribution', fontsize=14)
            plt.xlabel('Pixel Ratio', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, 'distribution_histogram.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 6.3 Box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df[column], color=PLOT_COLORS[column])
            plt.title(f'{column} Box Plot', fontsize=14)
            plt.ylabel('Pixel Ratio', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, 'box_plot.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 6.4 QQ plot
            plt.figure(figsize=(10, 6))
            stats.probplot(df[column], dist="norm", plot=plt)
            plt.title(f'{column} Q-Q Plot', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, 'qq_plot.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 6.5 Cumulative distribution
            plt.figure(figsize=(10, 6))
            sns.ecdfplot(data=df[column], color=PLOT_COLORS[column])
            plt.title(f'{column} Cumulative Distribution', fontsize=14)
            plt.xlabel('Pixel Ratio', fontsize=12)
            plt.ylabel('Cumulative Probability', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(class_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight', transparent=True)
            plt.close()
            
            # 6.6 Save class statistics
            class_stats = {
                'statistics': df[column].describe().to_dict(),
                'statistical_tests': {
                    'skewness': float(stats.skew(df[column])),
                    'kurtosis': float(stats.kurtosis(df[column])),
                    'normality_test': {
                        'statistic': float(stats.normaltest(df[column])[0]),
                        'p_value': float(stats.normaltest(df[column])[1])
                    }
                },
                'percentiles': {
                    '1%': float(np.percentile(df[column], 1)),
                    '5%': float(np.percentile(df[column], 5)),
                    '10%': float(np.percentile(df[column], 10)),
                    '25%': float(np.percentile(df[column], 25)),
                    '50%': float(np.percentile(df[column], 50)),
                    '75%': float(np.percentile(df[column], 75)),
                    '90%': float(np.percentile(df[column], 90)),
                    '95%': float(np.percentile(df[column], 95)),
                    '99%': float(np.percentile(df[column], 99))
                }
            }
            
            with open(os.path.join(class_dir, 'statistics.json'), 'w') as f:
                json.dump(class_stats, f, indent=4)
        
        # 7. Create 3D waterfall plot
        plt.figure(figsize=(15, 10))
        ax = plt.axes(projection='3d')
        
        # Create waterfall plots for each class
        x = np.arange(len(df.columns))
        y = np.linspace(0, 1, 100)  # Pixel ratio range
        
        for i, column in enumerate(df.columns):
            # Calculate kernel density estimate
            kde = stats.gaussian_kde(df[column])
            z = kde(y)
            
            # Create waterfall plot
            ax.plot(x[i] * np.ones_like(y), y, z, 
                   color=PLOT_COLORS[column], 
                   alpha=0.7,
                   label=column)
            
            # Create fill effect
            verts = [(x[i], y[j], z[j]) for j in range(len(y))]
            verts.append((x[i], y[-1], 0))  # Add bottom point
            verts.append((x[i], y[0], 0))   # Add starting point
            
            # Create polygon
            poly = Poly3DCollection([verts], alpha=0.3)
            poly.set_facecolor(PLOT_COLORS[column])
            ax.add_collection3d(poly)
        
        # Set plot attributes
        ax.set_xlabel('Classes')
        ax.set_ylabel('Pixel Ratio')
        ax.set_zlabel('Density')
        ax.set_title('3D Waterfall Plot of Class Distributions')
        
        # Set x-axis ticks
        ax.set_xticks(x)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_dir, '3d_waterfall_plot.png'), 
                   dpi=300, 
                   bbox_inches='tight', 
                   transparent=True)
        plt.close()
        
        # Save overall statistics
        statistics = self.generate_statistics()
        statistics['overall'].to_csv(os.path.join(stats_dir, 'overall_statistics.csv'))
        
        # Save summary information
        summary = {
            'total_images': len(self.distributions),
            'class_means': df.mean().to_dict(),
            'class_stds': df.std().to_dict(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(stats_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nAnalysis results saved to directory: {output_dir}")
        print("Generated files:")
        print("1. statistics/ - Statistical analysis results")
        print("   - Overall statistics and plots")
        print("   - Class-specific directories with:")
        print("     * Statistical analysis plots")
        print("     * Class-specific statistics")
        print("   - Summary information")

def main():
    # Set data folder path
    data_folder = "/ssd-data/jshen-data/VIGOR"  # Please modify according to actual path
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"vigor_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer instance
    analyzer = LabelAnalyzer(data_folder, num_workers=8)
    
    # Analyze all cities
    analyzer.analyze_all_cities()
    
    # Generate statistics
    stats = analyzer.generate_statistics()
    
    # Print basic statistics
    print("\nOverall Statistics:")
    print(stats['overall'])
    
    print("\nCity-wise Statistics:")
    for city, city_stats in stats['by_city'].items():
        print(f"\n{city}:")
        print(city_stats)
    
    # Plot distributions
    analyzer.plot_distributions(output_dir)

if __name__ == "__main__":
    main() 