# dsp_block/dsp.py
import cv2  # OpenCV2
import numpy as np
import json
import argparse
import sys
import os

# Handle arguments passed by Edge Impulse runner
parser = argparse.ArgumentParser(description="Custom Image Preprocessor(OpenCV)")
parser.add_argument(
    "--features-out",
    type=str,
    required=True,
    help="Output file for the features (e.g., features.npy)",
)
parser.add_argument(
    "--draw-graphs-out",
    type=str,
    required=True,
    help="Output directory for graphs (unused here)",
)
parser.add_argument(
    "--info-out",
    type=str,
    required=True,
    help="Output file for block info (e.g., info.json)",
)
parser.add_argument(
    "--in-file", type=str, required=True, help="Input file path (e.g., image.jpg)"
)

# Custom arguments from parameters.json
parser.add_argument("--img_width", type=int, required=True, help="Target image width")
parser.add_argument("--img_height", type=int, required=True, help="Target image height")

args, unknown = parser.parse_known_args()

# --- Image Processing ---
try:
    # 1. Load the image using OpenCV
    # cv2.IMREAD_COLOR loads as BGR by default
    img = cv2.imread(args.in_file, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image file: {args.in_file}")

    print(f"Original image shape: {img.shape}")

    # 2. Resize the image to the target dimensions
    target_size = (args.img_width, args.img_height)
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    print(f"Resized image shape: {img_resized.shape}")

    # 3. Convert from BGR (OpenCV default) to RGB (standard for ML)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 4. Scale pixel values from 0-255 to 0-1 (as floating point)
    img_scaled = img_rgb.astype(np.float32) / 255.0

    # --- Output Features ---
    # The features are the processed image array itself
    features = img_scaled

    # Save the features as a NumPy file
    print(f"Saving features with shape {features.shape} to {args.features_out}")
    np.save(args.features_out, features)

    # --- Output Block Info ---
    # Create a basic info.json file (required by EI)
    info = {
        "success": True,
        "can_visualize": False,  # We are not generating a graph output
        "warnings": [],
    }
    with open(args.info_out, "w") as f:
        json.dump(info, f)

    print("DSP block finished successfully.")
    sys.exit(0)

except Exception as e:
    print(f"Error processing image {args.in_file}: {e}", file=sys.stderr)

    # Output error info
    info = {"success": False, "error": str(e), "warnings": []}
    try:
        with open(args.info_out, "w") as f:
            json.dump(info, f)
    except Exception as write_e:
        print(f"Failed to write error info: {write_e}", file=sys.stderr)

    sys.exit(1)
