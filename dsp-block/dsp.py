# dsp_block/dsp.py
import cv2  # OpenCV
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
    help="Output file for features (e.g., features.npy)",
)
parser.add_argument(
    "--draw-graphs-out",
    type=str,
    help="Output directory for graphs (unused here)",
)
parser.add_argument(
    "--info-out",
    type=str,
    help="Output file for block info (e.g., info.json)",
)
parser.add_argument("--in-file", type=str, help="Input file path (e.g., image.jpg)")

# Custom arguments from parameters.json
parser.add_argument("--img_width", type=int, help="Target image width")
parser.add_argument("--img_height", type=int, help="Target image height")

args, unknown = parser.parse_known_args()

# --- MANUAL Argument Check ---
required_args_for_processing = [
    "features_out",
    "in_file",
    "img_width",
    "img_height",
    "info_out",
    "draw_graphs_out",
]
missing_args = [
    arg
    for arg in required_args_for_processing
    if getattr(args, arg.replace("-", "_")) is None
]

# Check if --in-file is provided. If not, assume it's a validation check.
if args.in_file is None:
    print("Validation check detected (missing --in-file). Attempting clean exit.")
    info = {
        "success": True,
        "warnings": ["Validation check mode - no processing performed."],
    }

    # Define default info_out path if not provided by runner
    info_out_path = (
        args.info_out if args.info_out else "./info.json"
    )  # Default to current dir

    # ALWAYS try to write info.json
    try:
        with open(info_out_path, "w") as f:
            json.dump(info, f)
        print(f"Successfully wrote validation info to {info_out_path}")
    except Exception as write_e:
        # If writing fails even during validation, print error but still exit cleanly
        print(
            f"Warning: Failed to write info.json during validation check: {write_e}",
            file=sys.stderr,
        )

    # Exit cleanly regardless of whether info.json could be written
    sys.exit(0)

# If --in-file *is* provided, it's a real processing job. Check ALL required args.
elif missing_args:
    # ... (error handling for real job - check info_out exists before writing) ...
    error_message = f"Error: Real processing job detected, but missing required arguments: {', '.join(missing_args)}"
    print(error_message, file=sys.stderr)
    if args.info_out:  # Check if path exists
        info = {"success": False, "error": error_message, "warnings": []}
        try:
            with open(args.info_out, "w") as f:
                json.dump(info, f)
        except Exception as write_e:
            print(f"Failed to write error info: {write_e}", file=sys.stderr)
    else:
        print(
            "Error: Cannot write error status because --info-out is missing.",
            file=sys.stderr,
        )
    sys.exit(1)

# If --in-file is present AND no arguments are missing, proceed.
else:
    print("All required arguments found. Proceeding with image processing.")

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
