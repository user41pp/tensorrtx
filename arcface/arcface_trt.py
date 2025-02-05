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
import enum
from typing import Optional

# Constants
INPUT_H = 112  # Height of input images
INPUT_W = 112  # Width of input images
OUTPUT_SIZE = 512  # Size of the output feature vector
BATCH_SIZE = 1
DEVICE = 0  # GPU id

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Verbosity levels
class VerbosityLevel(enum.IntEnum):
    SILENT = 0  # No output
    ERROR = 1   # Only error messages
    INFO = 2    # Important high-level information
    DEBUG = 3   # Detailed debugging information
    TRACE = 4   # Very detailed tracing information

# Global verbosity setting
VERBOSITY = VerbosityLevel.INFO

def log(level: VerbosityLevel, message: str, error: Optional[Exception] = None) -> None:
    """
    Log a message if the current verbosity level is high enough.
    
    Args:
        level: The verbosity level of this message
        message: The message to log
        error: Optional exception to include in output
    """
    if level <= VERBOSITY:
        if error:
            print(f"{level.name}: {message}\nError: {str(error)}", file=sys.stderr if level == VerbosityLevel.ERROR else sys.stdout)
        else:
            print(f"{level.name}: {message}", file=sys.stderr if level == VerbosityLevel.ERROR else sys.stdout)

def normalize_image(img):
    """
    Normalize image for ArcFace model to match C++ preprocessing exactly
    """
    log(VerbosityLevel.DEBUG, f"Input image shape before resize: {img.shape}")
    log(VerbosityLevel.DEBUG, f"Input image data type: {img.dtype}")
    log(VerbosityLevel.DEBUG, f"Input image range: min={img.min()}, max={img.max()}")
    
    # Resize image to expected dimensions
    img = img[0:INPUT_H, 0:INPUT_W]
    log(VerbosityLevel.DEBUG, f"After resize: {img.shape}")
    log(VerbosityLevel.DEBUG, f"After resize range: min={img.min()}, max={img.max()}")
    
    # Convert to float32 and reshape to match C++ memory layout
    img_flat = img.reshape(-1, 3)
    log(VerbosityLevel.DEBUG, f"After flatten: {img_flat.shape}")
    
    # Sample some pixels for debugging
    log(VerbosityLevel.TRACE, "Sample pixels before normalization:")
    for i in range(0, INPUT_H * INPUT_W, (INPUT_H * INPUT_W) // 4):
        log(VerbosityLevel.TRACE, f"Pixel {i}: BGR={img_flat[i]}, RGB={img_flat[i][::-1]}")
        
    # Print exact values for first few pixels to match against C++
    log(VerbosityLevel.TRACE, "Detailed first few pixels (BGR):")
    for i in range(5):
        b, g, r = img_flat[i]
        log(VerbosityLevel.TRACE, f"Pixel {i}:")
        log(VerbosityLevel.TRACE, f"  B: {b} -> {(float(b) - 127.5) * 0.0078125:.6f}")
        log(VerbosityLevel.TRACE, f"  G: {g} -> {(float(g) - 127.5) * 0.0078125:.6f}")
        log(VerbosityLevel.TRACE, f"  R: {r} -> {(float(r) - 127.5) * 0.0078125:.6f}")
    
    # Allocate output array in CHW format
    out = np.empty((3, INPUT_H, INPUT_W), dtype=np.float32)
    log(VerbosityLevel.DEBUG, f"Output buffer shape: {out.shape}")
    
    # Process each pixel to match C++ exactly
    for i in range(INPUT_H * INPUT_W):
        out[0, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][2]) - 127.5) * 0.0078125  # R
        out[1, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][1]) - 127.5) * 0.0078125  # G
        out[2, i // INPUT_W, i % INPUT_W] = (float(img_flat[i][0]) - 127.5) * 0.0078125  # B
    
    # Add batch dimension
    out = np.expand_dims(out, axis=0)
    log(VerbosityLevel.DEBUG, f"Final shape with batch dimension: {out.shape}")
    log(VerbosityLevel.DEBUG, f"Data range: min={out.min():.6f}, max={out.max():.6f}")
    
    # Sample some normalized pixels for debugging
    log(VerbosityLevel.TRACE, "Sample normalized pixels (first 4 per channel):")
    for c, name in enumerate(['R', 'G', 'B']):
        log(VerbosityLevel.TRACE, f"Channel {name}: {out[0,c,0,:4]}")
    
    return np.ascontiguousarray(out)

class ArcFace_TRT(object):
    """
    description: A ArcFace class that warps TensorRT ops, preprocess and postprocess ops.
    """
    def __init__(self, engine_file_path):
        log(VerbosityLevel.INFO, "Initializing ArcFace_TRT...")
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        self._context_pushed = False
        self._context_cleaned_up = False
        self._resources_cleaned = False
        
        self.stream = cuda.Stream()
        log(VerbosityLevel.DEBUG, "Created CUDA stream")
        
        self.runtime = trt.Runtime(TRT_LOGGER)
        log(VerbosityLevel.DEBUG, "Created TensorRT runtime")

        log(VerbosityLevel.INFO, f"Loading engine from: {engine_file_path}")
        with open(engine_file_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        log(VerbosityLevel.INFO, "Engine deserialized successfully")

        self.context = self.engine.create_execution_context()
        log(VerbosityLevel.DEBUG, "Created execution context")

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self._allocated_buffers = []

        for binding in self.engine:
            shape = self.engine.get_tensor_shape(binding)
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_size = trt.volume(shape)
                log(VerbosityLevel.DEBUG, f"TensorRT engine input shape: {shape}")
                log(VerbosityLevel.DEBUG, f"Input size in elements: {self.input_size}")
            else:
                self.output_shape = shape
                self.output_size = trt.volume(shape)
                log(VerbosityLevel.DEBUG, f"TensorRT engine output shape: {shape}")
                log(VerbosityLevel.DEBUG, f"Output size in elements: {self.output_size}")
            
            host_mem = cuda.pagelocked_empty(
                self.input_size if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT else self.output_size,
                trt.nptype(self.engine.get_tensor_dtype(binding))
            )
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self._allocated_buffers.append(cuda_mem)
            self.bindings.append(int(cuda_mem))
            
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        log(VerbosityLevel.INFO, "Initialization completed successfully")

    def cleanup_cuda_resources(self):
        log(VerbosityLevel.DEBUG, "Starting CUDA resource cleanup...")
        if hasattr(self, '_resources_cleaned') and self._resources_cleaned:
            log(VerbosityLevel.DEBUG, "Resources already cleaned up")
            return

        try:
            if hasattr(self, 'stream'):
                log(VerbosityLevel.DEBUG, "Synchronizing CUDA stream...")
                self.stream.synchronize()
                log(VerbosityLevel.DEBUG, "Stream synchronized")

            if hasattr(self, 'context'):
                delattr(self, 'context')

            if hasattr(self, 'engine'):
                delattr(self, 'engine')

            if hasattr(self, '_allocated_buffers'):
                log(VerbosityLevel.DEBUG, f"Freeing {len(self._allocated_buffers)} CUDA buffers...")
                for i, buf in enumerate(self._allocated_buffers):
                    try:
                        buf.free()
                        log(VerbosityLevel.TRACE, f"Freed buffer {i}")
                    except Exception as e:
                        log(VerbosityLevel.ERROR, f"Error freeing buffer {i}", e)
                self._allocated_buffers = []

            if hasattr(self, 'host_inputs'):
                self.host_inputs = []
            if hasattr(self, 'host_outputs'):
                self.host_outputs = []
            if hasattr(self, 'cuda_inputs'):
                self.cuda_inputs = []
            if hasattr(self, 'cuda_outputs'):
                self.cuda_outputs = []
            if hasattr(self, 'bindings'):
                self.bindings = []

            if hasattr(self, 'stream'):
                log(VerbosityLevel.DEBUG, "Clearing CUDA stream reference...")
                delattr(self, 'stream')
                log(VerbosityLevel.DEBUG, "Stream reference cleared")

            if hasattr(self, 'runtime'):
                log(VerbosityLevel.DEBUG, "Clearing TensorRT runtime reference...")
                delattr(self, 'runtime')
                log(VerbosityLevel.DEBUG, "Runtime reference cleared")

            self._resources_cleaned = True
            log(VerbosityLevel.INFO, "CUDA resource cleanup completed")

        except Exception as e:
            log(VerbosityLevel.ERROR, "Error during resource cleanup", e)

    def infer(self, image_path, is_first=True):
        try:
            log(VerbosityLevel.INFO, "Starting inference...")
            if not self._context_pushed:
                self.cfx.push()
                self._context_pushed = True
            log(VerbosityLevel.DEBUG, "CUDA context pushed")
            
            log(VerbosityLevel.INFO, f"Loading image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            input_image = normalize_image(image)
            log(VerbosityLevel.DEBUG, "Image normalized successfully")
            
            if input_image.size != self.input_size:
                raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {input_image.size}\n"
                               f"Input shape: {input_image.shape}, Expected shape: {self.input_shape}")
            
            log(VerbosityLevel.DEBUG, "Copying data to host buffer...")
            np.copyto(self.host_inputs[0], input_image.ravel())
            log(VerbosityLevel.DEBUG, "Data copied to host buffer")
            
            log(VerbosityLevel.DEBUG, "Transferring data to GPU...")
            cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            log(VerbosityLevel.DEBUG, "Data transferred to GPU")
            
            log(VerbosityLevel.DEBUG, "Preparing for inference...")
            for binding_idx, binding in enumerate(self.engine):
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    log(VerbosityLevel.DEBUG, f"Setting input shape for binding {binding_idx}: {self.input_shape}")
                    self.context.set_input_shape(binding, self.input_shape)
            
            log(VerbosityLevel.INFO, "Running inference...")
            self.context.execute_async_v2(bindings=self.bindings,
                                        stream_handle=self.stream.handle)
            log(VerbosityLevel.INFO, "Inference completed")
            
            log(VerbosityLevel.DEBUG, "Transferring results back to host...")
            cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
            log(VerbosityLevel.DEBUG, "Results transferred")
            
            log(VerbosityLevel.DEBUG, "Synchronizing CUDA stream...")
            self.stream.synchronize()
            log(VerbosityLevel.DEBUG, "Stream synchronized")
            
            output = self.host_outputs[0]
            log(VerbosityLevel.DEBUG, f"Raw output shape: {output.shape}")
            log(VerbosityLevel.DEBUG, f"Raw output range: min={output.min():.6f}, max={output.max():.6f}")
            log(VerbosityLevel.TRACE, f"First 10 raw output values:\n{output[:10]}")
            
            feature = output.reshape(-1)[:OUTPUT_SIZE]
            log(VerbosityLevel.DEBUG, f"Reshaped feature vector shape: {feature.shape}")
            log(VerbosityLevel.DEBUG, f"Feature vector range: min={feature.min():.6f}, max={feature.max():.6f}")
            
            log(VerbosityLevel.TRACE, f"First 10 feature values before normalization:\n{feature[:10]}")
            
            if is_first:
                feature_mat = feature.reshape(512, 1).astype(np.float32)
            else:
                feature_mat = feature.reshape(1, 512).astype(np.float32)
                
            log(VerbosityLevel.DEBUG, f"Feature matrix shape before normalization: {feature_mat.shape}")
            feature_norm = cv2.normalize(feature_mat, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L2)
            log(VerbosityLevel.DEBUG, f"Feature matrix shape after normalization: {feature_norm.shape}")
            
            log(VerbosityLevel.DEBUG, f"Normalized feature matrix range: min={feature_norm.min():.6f}, max={feature_norm.max():.6f}")
            log(VerbosityLevel.TRACE, f"First 10 normalized values:\n{feature_norm.flatten()[:10]}")
            
            return feature_norm
            
        except Exception as e:
            log(VerbosityLevel.ERROR, "Error during inference", e)
            raise
        finally:
            log(VerbosityLevel.DEBUG, "Cleaning up CUDA context in infer...")
            try:
                if self._context_pushed and not self._context_cleaned_up:
                    self.cfx.pop()
                    self._context_pushed = False
                    log(VerbosityLevel.DEBUG, "CUDA context popped in infer")
                else:
                    log(VerbosityLevel.DEBUG, "No context to pop in infer")
            except Exception as e:
                log(VerbosityLevel.ERROR, "Error during CUDA cleanup in infer", e)

    def __del__(self):
        log(VerbosityLevel.DEBUG, "Starting ArcFace_TRT cleanup...")
        try:
            if hasattr(self, 'cfx') and not self._context_cleaned_up:
                log(VerbosityLevel.DEBUG, "Starting cleanup sequence...")
                
                self.cleanup_cuda_resources()
                
                if self._context_pushed:
                    log(VerbosityLevel.DEBUG, "Popping CUDA context...")
                    try:
                        self.cfx.pop()
                        self._context_pushed = False
                        log(VerbosityLevel.DEBUG, "Context popped successfully")
                    except Exception as e:
                        log(VerbosityLevel.ERROR, "Error popping context", e)
                
                log(VerbosityLevel.DEBUG, "Detaching CUDA context...")
                try:
                    self.cfx.detach()
                    log(VerbosityLevel.DEBUG, "Context detached successfully")
                except Exception as e:
                    log(VerbosityLevel.ERROR, "Error detaching context", e)
                finally:
                    delattr(self, 'cfx')
                
                self._context_cleaned_up = True
                log(VerbosityLevel.INFO, "Cleanup sequence completed")
            else:
                log(VerbosityLevel.DEBUG, "No CUDA context to clean up or already cleaned")
        except Exception as e:
            log(VerbosityLevel.ERROR, "Error during cleanup", e)
        finally:
            log(VerbosityLevel.DEBUG, "Cleanup finished")

def compute_similarity(feature1_mat, feature2_mat):
    log(VerbosityLevel.DEBUG, "Computing similarity between feature matrices:")
    log(VerbosityLevel.DEBUG, f"Feature 1 matrix shape: {feature1_mat.shape}")
    log(VerbosityLevel.DEBUG, f"Feature 2 matrix shape: {feature2_mat.shape}")
    
    log(VerbosityLevel.TRACE, f"First 5 values of feature1:\n{feature1_mat.flatten()[:5]}")
    log(VerbosityLevel.TRACE, f"First 5 values of feature2:\n{feature2_mat.flatten()[:5]}")
    
    log(VerbosityLevel.DEBUG, f"Feature 1 L2 norm: {np.linalg.norm(feature1_mat):.6f}")
    log(VerbosityLevel.DEBUG, f"Feature 2 L2 norm: {np.linalg.norm(feature2_mat):.6f}")
    
    result = np.matmul(feature2_mat, feature1_mat)
    similarity = result.item()
    log(VerbosityLevel.INFO, f"Matrix multiplication result: {similarity:.6f}")
    
    return similarity

def main():
    parser = argparse.ArgumentParser(description='ArcFace inference with TensorRT')
    parser.add_argument('reference', help='Path to reference face image')
    parser.add_argument('input', help='Path to input face image to compare against reference')
    parser.add_argument('--plugin', default="build/libarcface.so",
                      help='Path to TensorRT plugin library (default: build/libarcface.so)')
    parser.add_argument('--engine', default="build/arcface-r100.engine",
                      help='Path to TensorRT engine file (default: build/arcface-r100.engine)')
    parser.add_argument('--verbosity', type=int, choices=range(5), default=2,
                      help='Verbosity level (0=SILENT, 1=ERROR, 2=INFO, 3=DEBUG, 4=TRACE)')
    args = parser.parse_args()

    # Set global verbosity level
    global VERBOSITY
    VERBOSITY = VerbosityLevel(args.verbosity)

    # Load custom plugins
    if not os.path.exists(args.plugin):
        log(VerbosityLevel.ERROR, f"Plugin library not found: {args.plugin}")
        sys.exit(1)
    ctypes.CDLL(args.plugin)

    # Check engine file exists
    if not os.path.exists(args.engine):
        log(VerbosityLevel.ERROR, f"Engine file not found: {args.engine}")
        sys.exit(1)

    # Check input files exist
    if not os.path.exists(args.reference):
        log(VerbosityLevel.ERROR, f"Reference image not found: {args.reference}")
        sys.exit(1)
    if not os.path.exists(args.input):
        log(VerbosityLevel.ERROR, f"Input image not found: {args.input}")
        sys.exit(1)

    arcface = None
    try:
        arcface = ArcFace_TRT(args.engine)

        log(VerbosityLevel.INFO, f"Processing reference image: {args.reference}")
        ref_feature = arcface.infer(args.reference, is_first=True)
        
        log(VerbosityLevel.INFO, f"Processing input image: {args.input}")
        input_feature = arcface.infer(args.input, is_first=False)

        similarity = compute_similarity(ref_feature, input_feature)
        log(VerbosityLevel.INFO, f"Similarity score: {similarity:.4f}")
        
    except Exception as e:
        log(VerbosityLevel.ERROR, "Error during inference", e)
        sys.exit(1)
    finally:
        if arcface:
            arcface.cleanup_cuda_resources()

    log(VerbosityLevel.INFO, "ArcFace_TRT inference completed")

if __name__ == "__main__":
    main()
