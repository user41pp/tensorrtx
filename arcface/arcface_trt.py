import ctypes
import os
import sys
import time
import argparse
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# Constants
INPUT_H = 112  # Height of input images
INPUT_W = 112  # Width of input images
OUTPUT_SIZE = 512  # Size of the output feature vector
BATCH_SIZE = 1
DEVICE = 0  # GPU id

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def normalize_image(img):
    """
    Normalize image for ArcFace model to match C++ preprocessing exactly
    """
    print(f"\nInput image shape before resize: {img.shape}")
    print(f"Input image data type: {img.dtype}")
    print(f"Input image range: min={img.min()}, max={img.max()}")
    
    # Resize image to expected dimensions
    #img = cv2.resize(img, (INPUT_W, INPUT_H))
    img = img[0:INPUT_H, 0:INPUT_W]
    print(f"After resize: {img.shape}")
    print(f"After resize range: min={img.min()}, max={img.max()}")
    
    # Convert to float32 and reshape to match C++ memory layout
    img_flat = img.reshape(-1, 3)
    print(f"After flatten: {img_flat.shape}")
    
    # Sample some pixels for debugging
    print("\nSample pixels before normalization:")
    for i in range(0, INPUT_H * INPUT_W, (INPUT_H * INPUT_W) // 4):
        print(f"Pixel {i}: BGR={img_flat[i]}, RGB={img_flat[i][::-1]}")
        
    # Print exact values for first few pixels to match against C++
    print("\nDetailed first few pixels (BGR):")
    for i in range(5):
        b, g, r = img_flat[i]
        print(f"Pixel {i}:")
        print(f"  B: {b} -> {(float(b) - 127.5) * 0.0078125:.6f}")
        print(f"  G: {g} -> {(float(g) - 127.5) * 0.0078125:.6f}")
        print(f"  R: {r} -> {(float(r) - 127.5) * 0.0078125:.6f}")
    
    # Allocate output array in CHW format
    out = np.empty((3, INPUT_H, INPUT_W), dtype=np.float32)
    print(f"\nOutput buffer shape: {out.shape}")
    
    # Process each pixel to match C++ exactly
    for i in range(INPUT_H * INPUT_W):
        # Note: img_flat[i] is in BGR order from OpenCV
        # Map to same channel order as C++: R,G,B
        out[0, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][2]) - 127.5) * 0.0078125  # R
        out[1, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][1]) - 127.5) * 0.0078125  # G
        out[2, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][0]) - 127.5) * 0.0078125  # B
    
    # Add batch dimension
    out = np.expand_dims(out, axis=0)
    print(f"Final shape with batch dimension: {out.shape}")
    print(f"Data range: min={out.min():.6f}, max={out.max():.6f}")
    
    # Sample some normalized pixels for debugging
    print("\nSample normalized pixels (first 4 per channel):")
    for c, name in enumerate(['R', 'G', 'B']):
        print(f"Channel {name}: {out[0,c,0,:4]}")
    
    return np.ascontiguousarray(out)

class ArcFace_TRT(object):
    """
    description: A ArcFace class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_file_path):
        # Create a Context on this device,
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            # Get shape and make it compatible with explicit batch
            shape = engine.get_tensor_shape(binding)
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_size = trt.volume(shape)
                print(f"\nTensorRT engine input shape: {shape}")
                print(f"Input size in elements: {self.input_size}")
            else:
                self.output_shape = shape
                self.output_size = trt.volume(shape)
                print(f"TensorRT engine output shape: {shape}")
                print(f"Output size in elements: {self.output_size}")
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(
                self.input_size if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT else self.output_size,
                trt.nptype(engine.get_tensor_dtype(binding))
            )
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def infer(self, image_path, is_first=True):
        try:
            print("\nStarting inference...")
            self.cfx.push()
            self._context_pushed = True
            print("CUDA context pushed")
            
            # Process input image
            print(f"Loading image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            input_image = normalize_image(image)
            print("Image normalized successfully")
            
            # Verify input shape matches expected shape
            if input_image.size != self.input_size:
                raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {input_image.size}\n"
                               f"Input shape: {input_image.shape}, Expected shape: {self.input_shape}")
            
            print("\nCopying data to host buffer...")
            # Copy input image to host buffer
            np.copyto(self.host_inputs[0], input_image.ravel())
            print("Data copied to host buffer")
            
            print("Transferring data to GPU...")
            # Transfer input data to the GPU
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            print("Data transferred to GPU")
            
            # Run inference
            print("\nPreparing for inference...")
            # Set input tensor shape
            for binding_idx, binding in enumerate(self.engine):
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    print(f"Setting input shape for binding {binding_idx}: {self.input_shape}")
                    self.context.set_input_shape(binding, self.input_shape)
            
            print("Running inference...")
            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings,
                                        stream_handle=self.stream.handle)
            print("Inference completed")
            
            print("\nTransferring results back to host...")
            # Transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            print("Results transferred")
            
            print("Synchronizing CUDA stream...")
            # Synchronize the stream
            self.stream.synchronize()
            print("Stream synchronized")
            
            # Get output and normalize
            output = self.host_outputs[0]
            print(f"\nRaw output shape: {output.shape}")
            print(f"Raw output range: min={output.min():.6f}, max={output.max():.6f}")
            print("\nFirst 10 raw output values:")
            print(output[:10])
            
            # Reshape to match C++ exactly - flatten all dimensions to get 512 elements
            feature = output.reshape(-1)[:OUTPUT_SIZE]  # Take first 512 elements
            print(f"\nReshaped feature vector shape: {feature.shape}")
            print(f"Feature vector range: min={feature.min():.6f}, max={feature.max():.6f}")
            
            # Sample some feature values
            print("\nFirst 10 feature values before normalization:")
            print(feature[:10])
            
            # Convert to OpenCV Mat format for exact normalization
            # Match C++ matrix shapes exactly:
            # First vector: (512, 1) column vector
            # Second vector: (1, 512) row vector
            if is_first:
                feature_mat = feature.reshape(512, 1).astype(np.float32)
            else:
                feature_mat = feature.reshape(1, 512).astype(np.float32)
                
            print(f"\nFeature matrix shape before normalization: {feature_mat.shape}")
            # Use cv2.NORM_L2 to match C++ normalization
            feature_norm = cv2.normalize(feature_mat, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L2)
            print(f"Feature matrix shape after normalization: {feature_norm.shape}")
            
            # Print normalized values for debugging
            print(f"\nNormalized feature matrix range: min={feature_norm.min():.6f}, max={feature_norm.max():.6f}")
            print("First 10 normalized values:")
            print(feature_norm.flatten()[:10])
            
            return feature_norm
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            print("\nCleaning up CUDA context in infer...")
            try:
                if hasattr(self, '_context_pushed'):
                    self.cfx.pop()
                    print("CUDA context popped in infer")
                else:
                    print("No context to pop in infer")
            except Exception as e:
                print(f"Error during CUDA cleanup in infer: {str(e)}")

    def __del__(self):
        """Ensure context is removed even if an error occurs"""
        print("\nStarting ArcFace_TRT cleanup...")
        try:
            if hasattr(self, 'cfx') and not hasattr(self, '_context_cleaned_up'):
                print("Cleaning up CUDA context in destructor...")
                try:
                    self.cfx.pop()
                    print("Context popped successfully")
                except Exception as e:
                    print(f"Error popping context: {str(e)}")
                
                try:
                    self.cfx.detach()
                    print("Context detached successfully")
                except Exception as e:
                    print(f"Error detaching context: {str(e)}")
                
                self._context_cleaned_up = True
            else:
                print("No CUDA context to clean up or already cleaned")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            import traceback
            traceback.print_exc()

def compute_similarity(feature1_mat, feature2_mat):
    """
    Compute cosine similarity between two feature vectors using matrix multiplication
    exactly like the C++ code
    """
    # Print detailed debugging info
    print("\nComputing similarity between feature matrices:")
    print(f"Feature 1 matrix shape: {feature1_mat.shape}")
    print(f"Feature 2 matrix shape: {feature2_mat.shape}")
    
    # Print first few values of each matrix
    print("\nFirst 5 values of feature1:")
    print(feature1_mat.flatten()[:5])
    print("\nFirst 5 values of feature2:")
    print(feature2_mat.flatten()[:5])
    
    # Verify normalization
    print(f"\nFeature 1 L2 norm: {np.linalg.norm(feature1_mat):.6f}")
    print(f"Feature 2 L2 norm: {np.linalg.norm(feature2_mat):.6f}")
    
    # Compute similarity using matrix multiplication
    # In C++: cv::Mat res = out_norm1 * out_norm;
    result = np.matmul(feature2_mat, feature1_mat)
    similarity = result.item()  # Get scalar value
    print(f"\nMatrix multiplication result: {similarity:.6f}")
    
    return similarity

def main():
    parser = argparse.ArgumentParser(description='ArcFace inference with TensorRT')
    parser.add_argument('reference', help='Path to reference face image')
    parser.add_argument('input', help='Path to input face image to compare against reference')
    parser.add_argument('--plugin', default="build/libarcface.so",
                      help='Path to TensorRT plugin library (default: build/libarcface.so)')
    parser.add_argument('--engine', default="build/arcface-r100.engine",
                      help='Path to TensorRT engine file (default: build/arcface-r100.engine)')
    args = parser.parse_args()

    # Load custom plugins
    if not os.path.exists(args.plugin):
        raise FileNotFoundError(f"Plugin library not found: {args.plugin}")
    ctypes.CDLL(args.plugin)

    # Check engine file exists
    if not os.path.exists(args.engine):
        raise FileNotFoundError(f"Engine file not found: {args.engine}")

    # Check input files exist
    if not os.path.exists(args.reference):
        raise FileNotFoundError(f"Reference image not found: {args.reference}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    # Initialize ArcFace
    arcface = None
    try:
        arcface = ArcFace_TRT(args.engine)

        # Extract features from reference and input images
        print(f"Processing reference image: {args.reference}")
        ref_feature = arcface.infer(args.reference, is_first=True)  # Column vector
        
        print(f"Processing input image: {args.input}")
        input_feature = arcface.infer(args.input, is_first=False)  # Row vector

        # Compute similarity score using OpenCV matrices
        similarity = compute_similarity(ref_feature, input_feature)
        print(f"\nSimilarity score: {similarity:.4f}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        if arcface:
            arcface.__del__()
        # Clean up CUDA context
        try:
            cuda.Context.pop()
        except:
            pass

if __name__ == "__main__":
    main()
