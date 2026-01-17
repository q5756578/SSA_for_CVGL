"""
Training Log Parser and Analysis Tool

This script provides functionality to parse training logs from the VIGOR model training process,
extract key metrics and statistics, and generate comprehensive analysis reports. It can:
1. Parse training logs to extract metrics like loss, learning rate, and evaluation scores
2. Generate CSV and Excel reports with detailed statistics
3. Create visualizations of training metrics
4. Provide trend analysis of key metrics over training epochs

The script handles various metrics including:
- Training loss and learning rate
- Dataset statistics (pair pool, lengths, samples)
- Evaluation metrics (Recall@K, Hit Rate)
- Neighbour range expansion information
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_training_log(log_file):
    """
    Parse a training log file and extract training metrics and statistics.
    
    Args:
        log_file (str): Path to the training log file
        
    Returns:
        pd.DataFrame: DataFrame containing all extracted metrics and statistics
        
    The function extracts:
    - Epoch information
    - Training loss and learning rate
    - Dataset statistics (pair pool, lengths)
    - Evaluation metrics (if available)
    - Neighbour range expansion information
    - Sample addition statistics
    """
    # Initialize data list to store extracted information
    data = []
    
    # Read the log file
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Split content into epoch blocks
    epoch_blocks = content.split('------------------------------[Epoch:')
    
    # Parse each epoch block
    for block in epoch_blocks[1:]:  # Skip the first empty block
        try:
            # Extract epoch number
            epoch_num = int(re.search(r'(\d+)', block).group(1))
            
            # Extract training loss
            train_loss = float(re.search(r'Train Loss = ([\d.]+)', block).group(1))
            
            # Extract learning rate
            lr = float(re.search(r'Lr = ([\d.]+)', block).group(1))
            
            # Extract dataset information
            pair_pool = int(re.search(r'pair_pool: (\d+)', block).group(1))
            original_length = int(re.search(r'Original Length: (\d+)', block).group(1))
            length_after_shuffle = int(re.search(r'Length after Shuffle: (\d+)', block).group(1))
            pairs_left = int(re.search(r'Pairs left out of last batch to avoid creating noise: (\d+)', block).group(1))
            
            # Extract neighbour range expansion information
            neighbour_range_from = None
            neighbour_range_to = None
            if 'Expanding neighbour_range' in block:
                neighbour_range_match = re.search(r'Expanding neighbour_range from (\d+) to (\d+)', block)
                if neighbour_range_match:
                    neighbour_range_from = int(neighbour_range_match.group(1))
                    neighbour_range_to = int(neighbour_range_match.group(2))
            
            # Extract new samples information
            new_samples = None
            total_samples = None
            if 'Added' in block and 'new samples' in block:
                new_samples_match = re.search(r'Added (\d+) new samples', block)
                if new_samples_match:
                    new_samples = int(new_samples_match.group(1))
            
            if 'Total samples now:' in block:
                total_samples_match = re.search(r'Total samples now: (\d+)', block)
                if total_samples_match:
                    total_samples = int(total_samples_match.group(1))
            
            # Extract evaluation metrics if available
            eval_metrics = {}
            if '[Evaluate]' in block:
                eval_section = block.split('[Evaluate]')[1]
                recall1 = float(re.search(r'Recall@1: ([\d.]+)', eval_section).group(1))
                recall5 = float(re.search(r'Recall@5: ([\d.]+)', eval_section).group(1))
                recall10 = float(re.search(r'Recall@10: ([\d.]+)', eval_section).group(1))
                recall_top1 = float(re.search(r'Recall@top1: ([\d.]+)', eval_section).group(1))
                hit_rate = float(re.search(r'Hit_Rate: ([\d.]+)', eval_section).group(1))
                
                eval_metrics = {
                    'Recall@1': recall1,
                    'Recall@5': recall5,
                    'Recall@10': recall10,
                    'Recall@top1': recall_top1,
                    'Hit_Rate': hit_rate
                }
            
            # Create data row with all extracted information
            row = {
                'Epoch': epoch_num,
                'Train_Loss': train_loss,
                'Learning_Rate': lr,
                'Pair_Pool': pair_pool,
                'Original_Length': original_length,
                'Length_After_Shuffle': length_after_shuffle,
                'Pairs_Left': pairs_left,
                'New_Samples': new_samples,
                'Total_Samples': total_samples,
                'Neighbour_Range_From': neighbour_range_from,
                'Neighbour_Range_To': neighbour_range_to
            }
            
            # Add evaluation metrics to the row
            row.update(eval_metrics)
            
            data.append(row)
            
        except Exception as e:
            print(f"Error parsing epoch block: {e}")
            continue
    
    # Create DataFrame from extracted data
    df = pd.DataFrame(data)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as CSV
    csv_output = f'training_stats_{timestamp}.csv'
    df.to_csv(csv_output, index=False)
    print(f"Training statistics saved to {csv_output}")
    
    # Save as Excel with multiple sheets
    excel_output = f'training_stats_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        # Save main data sheet
        df.to_excel(writer, sheet_name='Training_Stats', index=False)
        
        # Create statistics summary sheet
        stats_df = pd.DataFrame({
            'Metric': ['Total Epochs', 'Mean Train Loss', 'Min Train Loss', 'Max Train Loss',
                      'Mean Pair Pool', 'Mean Original Length', 'Mean Length After Shuffle',
                      'Mean Pairs Left', 'Mean New Samples', 'Mean Total Samples',
                      'Neighbour Range From', 'Neighbour Range To'],
            'Value': [
                len(df),
                df['Train_Loss'].mean(),
                df['Train_Loss'].min(),
                df['Train_Loss'].max(),
                df['Pair_Pool'].mean(),
                df['Original_Length'].mean(),
                df['Length_After_Shuffle'].mean(),
                df['Pairs_Left'].mean(),
                df['New_Samples'].mean(),
                df['Total_Samples'].mean(),
                df['Neighbour_Range_From'].iloc[0] if not df['Neighbour_Range_From'].isna().all() else None,
                df['Neighbour_Range_To'].iloc[0] if not df['Neighbour_Range_To'].isna().all() else None
            ]
        })
        
        # Add evaluation metrics statistics if available
        if 'Recall@1' in df.columns:
            eval_stats = pd.DataFrame({
                'Metric': ['Mean Recall@1', 'Mean Recall@5', 'Mean Recall@10',
                          'Mean Recall@top1', 'Mean Hit Rate'],
                'Value': [
                    df['Recall@1'].mean(),
                    df['Recall@5'].mean(),
                    df['Recall@10'].mean(),
                    df['Recall@top1'].mean(),
                    df['Hit_Rate'].mean()
                ]
            })
            stats_df = pd.concat([stats_df, eval_stats], ignore_index=True)
        
        # Save statistics sheet
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        # Create trend analysis sheet
        trend_df = pd.DataFrame({
            'Metric': ['Train Loss Trend', 'Length After Shuffle Trend', 'New Samples Trend', 'Total Samples Trend'],
            'Start Value': [
                df['Train_Loss'].iloc[0],
                df['Length_After_Shuffle'].iloc[0],
                df['New_Samples'].iloc[0],
                df['Total_Samples'].iloc[0]
            ],
            'End Value': [
                df['Train_Loss'].iloc[-1],
                df['Length_After_Shuffle'].iloc[-1],
                df['New_Samples'].iloc[-1],
                df['Total_Samples'].iloc[-1]
            ],
            'Change': [
                f"{((df['Train_Loss'].iloc[-1] - df['Train_Loss'].iloc[0]) / df['Train_Loss'].iloc[0] * 100):.2f}%",
                f"{((df['Length_After_Shuffle'].iloc[-1] - df['Length_After_Shuffle'].iloc[0]) / df['Length_After_Shuffle'].iloc[0] * 100):.2f}%",
                f"{((df['New_Samples'].iloc[-1] - df['New_Samples'].iloc[0]) / df['New_Samples'].iloc[0] * 100):.2f}%",
                f"{((df['Total_Samples'].iloc[-1] - df['Total_Samples'].iloc[0]) / df['Total_Samples'].iloc[0] * 100):.2f}%"
            ]
        })
        trend_df.to_excel(writer, sheet_name='Trend_Analysis', index=False)
    
    print(f"Excel file saved as {excel_output}")
    
    return df

def plot_length_after_shuffle(df, log_file):
    """
    Create and save a plot showing the length after shuffle over training epochs.
    
    Args:
        df (pd.DataFrame): DataFrame containing the training metrics
        log_file (str): Path to the original log file (used for output filename)
        
    The function creates a line plot with:
    - Epochs on x-axis
    - Length after shuffle on y-axis
    - Data points marked with values
    - Grid lines for better readability
    """
    # Set matplotlib style
    plt.style.use('default')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the curve
    ax.plot(df['Epoch'], df['Length_After_Shuffle'], 
            marker='o', markersize=4, linewidth=2, 
            color='#1f77b4', alpha=0.7)
    
    # Add title and labels
    ax.set_title('Length After Shuffle vs Epoch', fontsize=14, pad=15)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Length After Shuffle', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to integers
    ax.set_xticks(range(0, df['Epoch'].max() + 1, 5))
    
    # Add data point labels
    for x, y in zip(df['Epoch'], df['Length_After_Shuffle']):
        ax.text(x, y, f'{y}', ha='center', va='bottom', fontsize=8)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = log_file.replace('.log', '_length_after_shuffle.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Path to the training log file
    log_file = "/home/jshen/Sample4Geo-my_DP/vigor_same/convnext_base.fb_in22k_ft_in1k_384/04271404/train_04271404.log"
    
    # Parse the log file and get the DataFrame
    df = parse_training_log(log_file)
    
    # Create visualization of length after shuffle
    plot_length_after_shuffle(df, log_file)
    
    # Print summary statistics
    print("\nTraining Statistics Summary:")
    print(f"Total Epochs: {len(df)}")
    print("\nLoss Statistics:")
    print(df['Train_Loss'].describe())
    print("\nDataset Statistics:")
    print(df[['Pair_Pool', 'Original_Length', 'Length_After_Shuffle', 'Pairs_Left']].describe())
    
    # Print evaluation metrics if available
    if 'Recall@1' in df.columns:
        print("\nEvaluation Metrics Statistics:")
        print(df[['Recall@1', 'Recall@5', 'Recall@10', 'Recall@top1', 'Hit_Rate']].describe()) 