# utils/model_conversion.py
import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm

def representative_dataset_generator(dataset_path, input_size, num_samples=100):
    """Generate representative dataset for quantization calibration.
    
    Args:
        dataset_path: Path to calibration images
        input_size: Model input size (width, height)
        num_samples: Number of samples to use
        
    Yields:
        Normalized image tensors
    """
    # Get all image files
    img_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, file))
    
    # Limit to num_samples
    if len(img_files) > num_samples:
        import random
        img_files = random.sample(img_files, num_samples)
    
    print(f"Using {len(img_files)} images for calibration")
    
    for img_path in tqdm(img_files):
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        yield [img]

def convert_model_to_tflite(model_path, output_path, input_size=(224, 224), 
                           calibration_dataset_path=None, quantize=True):
    """Convert TensorFlow model to TFLite with optional INT8 quantization.
    
    Args:
        model_path: Path to saved TensorFlow model
        output_path: Path to save TFLite model
        input_size: Model input dimensions (width, height)
        calibration_dataset_path: Path to dataset for INT8 calibration
        quantize: Whether to apply INT8 quantization
        
    Returns:
        bool: Success status
    """
    try:
        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantize:
            if not calibration_dataset_path:
                raise ValueError("Calibration dataset path required for INT8 quantization")
                
            print("Applying INT8 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            # Set the representative dataset
            converter.representative_dataset = lambda: representative_dataset_generator(
                calibration_dataset_path, input_size)
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model converted and saved to {output_path} ({model_size_mb:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

def benchmark_tflite_model(model_path, sample_image_path, input_size=(224, 224), num_runs=50):
    """Benchmark TFLite model inference speed.
    
    Args:
        model_path: Path to TFLite model
        sample_image_path: Path to sample image for testing
        input_size: Model input dimensions
        num_runs: Number of inference runs for averaging
        
    Returns:
        dict: Benchmark results
    """
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess the sample image
    img = cv2.imread(sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Check if quantized
    is_quantized = input_details[0]['dtype'] == np.int8
    
    if is_quantized:
        # For INT8 models, we need to quantize the input
        input_scale = input_details[0]['quantization'][0]
        input_zero_point = input_details[0]['quantization'][1]
        img = img / input_scale + input_zero_point
        img = img.astype(np.int8)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    
    # Warm-up run
    interpreter.invoke()
    
    # Benchmark runs
    import time
    inference_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        interpreter.invoke()
        inference_times.append((time.time() - start_time) * 1000)  # ms
    
    # Get results
    avg_time = sum(inference_times) / len(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    
    # Get memory usage
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
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
    # Example usage
    convert_model_to_tflite(
        "models/weed_detector.h5",
        "models/weed_detector_int8.tflite",
        input_size=(224, 224),
        calibration_dataset_path="dataset/calibration",
        quantize=True
    )
    
    results = benchmark_tflite_model(
        "models/weed_detector_int8.tflite",
        "dataset/test/sample.jpg",
        input_size=(224, 224)
    )
    
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    print(f"Average inference time: {results['avg_inference_ms']:.2f} ms")
    print(f"FPS: {results['fps']:.1f}")