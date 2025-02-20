import os
import pandas as pd
import numpy as np
import openslide
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import f1_score
from matplotlib import colormaps  # Import matplotlib colormap
from matplotlib.colors import Normalize
import argparse

# Initialize a colormap (e.g., 'viridis', 'plasma', 'coolwarm', 'hot')
colormap = colormaps.get_cmap('jet')  # Use the 'jet' colormap for a heatmap effect
norm = Normalize(vmin=0, vmax=1)  # Normalize prob values to the range [0, 1]

def create_prediction_overlay(svs_file, df_result, thumbnail_size=(1024, 1024)):
    # Step 1: Load the SVS file
    slide = openslide.OpenSlide(svs_file)
    
    # Step 2: Generate a thumbnail of the SVS file
    thumbnail = slide.get_thumbnail(thumbnail_size)
    thumbnail = thumbnail.convert("RGBA")  # Ensure RGBA for transparency support
    
    # Step 3: Prepare a blank mask image the same size as the thumbnail
    mask = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))  # Transparent initially
    
    # Step 4: Iterate through the DataFrame to extract patch info and add to the mask
    draw = ImageDraw.Draw(mask)
    
    # Calculate downscale factor based on thumbnail size vs full resolution
    downscale_factor = max(slide.level_dimensions[0][0], slide.level_dimensions[0][1]) / thumbnail_size[0]
    
    for _, row in df_result.iterrows():
        # Parse the x, y coordinates from the filename
        filename = row['filename']
        coords = filename.split('/')[-1].split('_')
        x, y = int(coords[0]), int(coords[1])
        patch_size = int(coords[2].split('.')[0])
    
        # Downscale coordinates to match thumbnail size
        x_downscaled = int(x / downscale_factor)
        y_downscaled = int(y / downscale_factor)
        patch_size_downscaled = int(patch_size / downscale_factor)
    
        # Set color based on prediction label
        if 0 <= row['prob'] <= 1:  # Ensure prob is within [0, 1]
            # intensity = int(row['prob'] * 255)  # Scale prob to 0-255 for RGB values
            # color = (255, 0, 0, intensity)  # Red with opacity based on probability
            rgba = colormap(norm(row['prob']))  # Map prob to colormap
            r, g, b, a = [int(c * 255) for c in rgba]  # Convert to 0-255 range
            color = (r, g, b, 150)
            # color = tuple(int(c * 255) for c in rgba[:4])  # Convert RGBA to 0-255 range
        else:
            color = (0, 0, 0, 0)  # Fully transparent for invalid prob values
    
        # Draw the rectangle on the mask
        draw.rectangle([x_downscaled, y_downscaled, x_downscaled + patch_size_downscaled, y_downscaled + patch_size_downscaled], fill=color)
    
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
    combined = Image.alpha_composite(thumbnail, mask)
    return thumbnail, combined

def process_wsi(svs_file, df_results, output_dir, suffix, thumbnail_size):
    svs_name = os.path.splitext(os.path.basename(svs_file))[0]
    df_results_sub = df_results[df_results['svs_name'] == svs_name]

    data = df_results_sub[df_results_sub['label'] != -1]
    true_labels = data['label']
    probabilities = data['prob']
    predictions = data['prob']>0.5
    f1 = f1_score(true_labels, predictions, zero_division=1)*100
    
    if df_results_sub['label'].sum() == 0:
        return  # Skip if there are no relevant labels
    
    if not df_results_sub.empty:
        output_image_path = os.path.join(output_dir, '{}_{}_{}.png'.format(svs_name, suffix, '{:.2f}'.format(f1)))
        thumbnail, overlay = create_prediction_overlay(svs_file, df_results_sub, thumbnail_size)
        overlay.save(output_image_path, "PNG")
        return svs_name  # Return the processed svs_name for tracking
    
def main(args):
    thumbnail_size = (1024, 1024)
    wsi_dir = args.wsi_dir
    wsi_list = glob(wsi_dir)

    results_dir = args.results_dir
    output_dir = os.path.join(results_dir, 'visual')
    print(f"Results Directory: {results_dir}")

    results_list = glob(os.path.join(results_dir, '*_inst.csv'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process result CSV files
    df_results = []
    for result_path in results_list:
        df_results.append(pd.read_csv(result_path))
    df_results = pd.concat(df_results, axis=0)
    df_results = df_results.groupby('filename', as_index=False).mean()
    df_results['svs_name'] = df_results['filename'].map(lambda x: x.split('/')[0])
    
    num_workers = min(args.num_workers, len(wsi_list))  # Adjust thread count based on CPU cores or user input
        
    # Multithreaded processing of WSIs
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_wsi,
                wsi_list[i],
                df_results,
                output_dir,
                args.model_name,
                thumbnail_size
            ): i for i in range(len(wsi_list))
        }
        
        # Track progress with tqdm
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                # Uncomment to log processed results
                # if result:
                #     print(f"Processed: {result}")
            except Exception as e:
                print(f"Error processing file: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process WSIs and generate visual overlays.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model (e.g., smmile).")
    parser.add_argument('--wsi_dir', type=str, required=True, help="Directory containing the WSI files.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory containing the result CSV files.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers for parallel processing.")
    
    args = parser.parse_args()
    main(args)
