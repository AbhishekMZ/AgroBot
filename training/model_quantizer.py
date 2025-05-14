# training/model_quantizer.py
import tensorflow as tf
import numpy as np
import os
import argparse
import glob
import time
from PIL import Image

def representative_dataset_generator(dataset_path, img_size=(224, 224), num_samples=100):
    """Generate representative dataset for quantization calibration.
    
    Args:
        dataset_path: Path to calibration images
        img_size: Model input dimensions (height, width)
        num_samples: Maximum number of samples to use
        
    Yields:
        Input tensor samples for quantization calibration
    """
    print(f"Loading calibration images from {dataset_path}")
    
    # Get all image files
    img_extensions = ['jpg', 'jpeg', 'png']
    img_files = []
    
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(dataset_path, f"**/*.{ext}"), recursive=True))
    
    if not img_files:
        raise ValueError(f"No images found in {dataset_path}")
    
    print(f"Found {len(img_files)} images for calibration")
    
    # Limit to num_samples
    if len(img_files) > num_samples:
        import random
        random.seed(42)  # For reproducibility
        img_files = random.sample(img_files, num_samples)
        print(f"Using {len(img_files)} random images for calibration")
    
    # Process each image
    for img_path in img_files:
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            yield [img_array]
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

def quantize_model(args):
    """Convert and quantize model to TFLite format.
    
    Args:
        args: Command line arguments with quantization parameters
    """
    print(f"Loading model from {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load the model
    model = tf.keras.models.load_model(args.model_path)
    input_shape = model.input_shape[1:3]  # (height, width)
    
    print(f"Model input shape: {input_shape}")
    
    # Basic TFLite conversion (no quantization)
    if not args.quantize:
        print("Converting model to TFLite format (no quantization)...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the model
        with open(args.output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Model converted and saved to {args.output_path}")
        print(f"Model size: {os.path.getsize(args.output_path) / (1024 * 1024):.2f} MB")
        return
    
    # Initialize TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flag
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # INT8 quantization requires representative dataset
    if args.quantize_type == 'int8':
        print("Performing INT8 quantization...")
        
        # Set INT8 as the target type
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        # Set representative dataset
        converter.representative_dataset = lambda: representative_dataset_generator(
            args.calibration_dir, 
            img_size=input_shape,
            num_samples=args.num_samples
        )
    
    # Float16 quantization
    elif args.quantize_type == 'float16':
        print("Performing float16 quantization...")
        converter.target_spec.supported_types = [tf.float16]
    
    # Dynamic range quantization (default)
    else:
        print("Performing dynamic range quantization...")
    
    # Convert the model
    try:
        tflite_model = converter.convert()
        
        # Save the model
        with open(args.output_path, 'wb') as f:
            f.write(tflite_model)
            
        model_size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
        print(f"Model converted and saved to {args.output_path}")
        print(f"Model size: {model_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return

def benchmark_tflite_model(model_path, sample_image_path, num_runs=50):
    """Benchmark TFLite model inference speed.
    
    Args:
        model_path: Path to TFLite model
        sample_image_path: Path to sample image for testing
        num_runs: Number of inference runs for averaging
        
    Returns:
        dict: Benchmark results
    """
    print(f"Loading model for benchmarking: {model_path}")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get model input size
    input_shape = input_details[0]['shape'][1:3]  # (height, width)
    
    print(f"Model input shape: {input_shape}")
    print(f"Input type: {input_details[0]['dtype']}")
    
    # Load and preprocess the sample image
    try:
        img = Image.open(sample_image_path).convert('RGB')
        img = img.resize(input_shape)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error loading sample image: {e}")
        return
    
    # Check if quantized to INT8
    is_quantized = input_details[0]['dtype'] == np.int8
    
    if is_quantized:
        # For INT8 models, we need to quantize the input
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        img_array = img_array / input_scale + input_zero_point
        img_array = img_array.astype(np.int8)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Warm-up run
    print("Performing warm-up run...")
    interpreter.invoke()
    
    # Benchmark runs
    print(f"Benchmarking with {num_runs} runs...")
    inference_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)
        
        if i % 10 == 0:
            print(f"Run {i+1}/{num_runs}: {inference_time:.2f} ms")
    
    # Calculate statistics
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    # Get memory usage
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    # Print results
    print("\nBenchmark Results:")
    print(f"Model: {model_path}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Quantized: {is_quantized}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Min inference time: {min_time:.2f} ms")
    print(f"Max inference time: {max_time:.2f} ms")
    print(f"Throughput: {1000 / avg_time:.1f} FPS")
    
    return {
        "model_path": model_path,
        "model_size_mb": model_size_mb,
        "quantized": is_quantized,
        "avg_inference_ms": avg_time,
        "min_inference_ms": min_time,
        "max_inference_ms": max_time,
        "fps": 1000 / avg_time
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize model to TFLite format")
    
    # Subparsers for different functions
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for quantize command
    quantize_parser = subparsers.add_parser("quantize", help="Quantize model to TFLite format")
    quantize_parser.add_argument('--model_path', required=True, help='Path to Keras model (.h5)')
    quantize_parser.add_argument('--output_path', required=True, help='Path to save TFLite model')
    quantize_parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    quantize_parser.add_argument('--quantize_type', choices=['dynamic', 'float16', 'int8'], 
                              default='dynamic', help='Type of quantization')
    quantize_parser.add_argument('--calibration_dir', help='Directory with calibration images (required for INT8)')
    quantize_parser.add_argument('--num_samples', type=int, default=100, help='Number of calibration samples')
    
    # Parser for benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark TFLite model")
    benchmark_parser.add_argument('--model_path', required=True, help='Path to TFLite model')
    benchmark_parser.add_argument('--sample_image', required=True, help='Path to sample image')
    benchmark_parser.add_argument('--num_runs', type=int, default=50, help='Number of inference runs')
    
    args = parser.parse_args()
    
    if args.command == "quantize":
        if args.quantize_type == 'int8' and not args.calibration_dir:
            parser.error("--calibration_dir is required for INT8 quantization")
        quantize_model(args)
    
    elif args.command == "benchmark":
        benchmark_tflite_model(args.model_path, args.sample_image, args.num_runs)
    
    else:
        parser.print_help()