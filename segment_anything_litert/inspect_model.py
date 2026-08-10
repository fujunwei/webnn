"""Download and inspect TFLite SAM model tensor names and shapes."""
import urllib.request
import sys
import os

# TFLite flatbuffers schema parsing
# The TFLite model starts with a FlatBuffer
# We need to parse the SubGraph to get tensor info

# Simple TFLite parser using flatbuffers reflection
# We'll use a minimal approach - parse the TFLite model binary

import struct

def read_tflite_model(filepath):
    """Parse a TFLite model file and extract tensor information."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # TFLite models are flatbuffers
    # The root is at offset indicated by the first 4 bytes after the file identifier
    # We'll use a simpler approach: parse the flatbuffer manually

    # For inspection, we'll use the tflite schema
    # But since we don't have the generated code, let's try importing tensorflow
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("=" * 60)
        print(f"Model: {os.path.basename(filepath)}")
        print("=" * 60)

        print("\nINPUTS:")
        for inp in input_details:
            print(f"  name: {inp['name']}")
            print(f"  shape: {inp['shape']}")
            print(f"  dtype: {inp['dtype']}")
            print(f"  index: {inp['index']}")
            print()

        print("OUTPUTS:")
        for out in output_details:
            print(f"  name: {out['name']}")
            print(f"  shape: {out['shape']}")
            print(f"  dtype: {out['dtype']}")
            print(f"  index: {out['index']}")
            print()

        # Also print all tensor details
        tensor_details = interpreter.get_tensor_details()
        print(f"\nALL TENSORS ({len(tensor_details)} total):")
        for t in tensor_details:
            print(f"  [{t['index']}] {t['name']}: shape={t['shape']}, dtype={t['dtype']}")

        return input_details, output_details

    except ImportError:
        print("tensorflow not available, trying tensorflow-cpu or tflite-runtime...")
        raise


def download_model(url, filepath):
    """Download a model file with progress display."""
    if os.path.exists(filepath):
        print(f"Already exists: {filepath}")
        return

    print(f"Downloading {url} -> {filepath}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            sys.stdout.write(f"\r  {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, filepath, report_progress)
    print()


if __name__ == "__main__":
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    hf_base = "https://huggingface.co/qualcomm/Segment-Anything-Model/resolve/main"

    # Start with decoder (smallest)
    files = [
        ("SAMDecoder.tflite", f"{hf_base}/SAMDecoder.tflite"),
    ]

    for filename, url in files:
        filepath = os.path.join(models_dir, filename)
        try:
            download_model(url, filepath)
            read_tflite_model(filepath)
        except Exception as e:
            print(f"Error: {e}")
