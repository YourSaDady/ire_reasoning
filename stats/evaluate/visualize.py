# %% [markdown]
"""
# EBM Landscape Visualization Notebook

This notebook processes a JSONL file containing EBM landscape data and generates heatmap visualizations for each time step.
"""
# %%
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
os.chdir('/home/user/shiqi/yichuan/EBM/ire_reasoning/stats')

# %%
def create_heatmap(data, output_path, title=None):
    """
    Create and save a heatmap from 2D data
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Energy Value', rotation=-90, va="bottom")
    
    # Set labels
    ax.set_xlabel('Possible Values (num_classes)')
    ax.set_ylabel('Variable Position (k)')
    if title:
        ax.set_title(title)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

# %%
def process_jsonl_file(input_file, output_base_dir):
    """
    Process JSONL file and generate visualization folders
    """
    # Read JSONL file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        data = json.loads(line)
        time_step = data['time_step']
        
        # Create main output folder for this line
        line_folder = os.path.join(output_base_dir, f"line_{line_idx+1}")
        os.makedirs(line_folder, exist_ok=True)
        
        # Process each EBM
        for ebm_key in [k for k in data.keys() if k.startswith('ebm_')]:
            ebm_num = ebm_key.split('_')[1]
            ebm_data = data[ebm_key]
            
            # Create EBM subfolder
            ebm_folder = os.path.join(line_folder, f"ebm_{ebm_num}")
            os.makedirs(ebm_folder, exist_ok=True)
            
            # Process each time step
            for time_key, time_data in ebm_data.items():
                if 'landscape' in time_data:
                    landscape = np.array(time_data['landscape'])
                    output_path = os.path.join(ebm_folder, f"{time_key}_heatmap.png")
                    
                    title = (f"EBM {ebm_num} - {time_key}\n"
                            f"Avg Energy: {time_data.get('avg_energy', 'N/A'):.2f} | "
                            f"Avg Loss: {time_data.get('avg_loss', 'N/A'):.2f}")
                    
                    create_heatmap(landscape, output_path, title)

# %%
# Main execution
input_jsonl = "./evaluate/binary-subtraction_mlp_visual.jsonl"  # Replace with your input file path
output_dir = "./visualizations/binary-subtraction_mlp"  # Output base directory

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process the file
process_jsonl_file(input_jsonl, output_dir)

print(f"Visualizations saved to: {output_dir}")