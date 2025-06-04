#!/usr/bin/env python
"""Check CUDA and GPU setup for TensorFlow."""

import os
import subprocess


def check_nvidia_smi():
    """Check if nvidia-smi is available."""
    print("1. Checking NVIDIA driver...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✓ NVIDIA driver is installed")
            print("\nGPU Info:")
            # Parse basic info
            lines = result.stdout.split("\n")
            for line in lines:
                if "NVIDIA" in line and "Driver Version" in line:
                    print(f"  {line.strip()}")
        else:
            print("✗ nvidia-smi not found. Please install NVIDIA drivers.")
            return False
    except Exception as e:
        print(f"✗ Error checking NVIDIA driver: {e}")
        return False
    return True


def check_cuda_paths():
    """Check CUDA installation paths."""
    print("\n2. Checking CUDA installation...")

    # Common CUDA paths on Windows
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\Program Files\NVIDIA Corporation\CUDA",
        r"C:\CUDA",
    ]

    cuda_found = False
    cuda_version = None

    for base_path in cuda_paths:
        if os.path.exists(base_path):
            # Look for versioned CUDA installations
            for item in os.listdir(base_path):
                if item.startswith("v"):
                    cuda_path = os.path.join(base_path, item)
                    if os.path.exists(cuda_path):
                        cuda_found = True
                        cuda_version = item
                        print(f"✓ Found CUDA installation: {cuda_path}")

                        # Check for important files
                        cudart_dll = os.path.join(cuda_path, "bin", "cudart64_12.dll")
                        if os.path.exists(cudart_dll):
                            print("  ✓ Found cudart64_12.dll")
                        else:
                            print("  ✗ Missing cudart64_12.dll (expected for CUDA 12.x)")

    if not cuda_found:
        print("✗ CUDA not found in standard locations")
        print("\nTo install CUDA 12.5:")
        print("1. Visit: https://developer.nvidia.com/cuda-12-5-0-download-archive")
        print("2. Download and install CUDA Toolkit 12.5")

    return cuda_found, cuda_version


def check_cudnn():
    """Check cuDNN installation."""
    print("\n3. Checking cuDNN...")

    # Check in CUDA directory
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    ]

    cudnn_found = False
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            cudnn_dll = os.path.join(cuda_path, "bin", "cudnn64_9.dll")
            if os.path.exists(cudnn_dll):
                cudnn_found = True
                print(f"✓ Found cuDNN: {cudnn_dll}")
                break

    if not cudnn_found:
        print("✗ cuDNN not found")
        print("\nTo install cuDNN 9.5:")
        print("1. Visit: https://developer.nvidia.com/cudnn")
        print("2. Download cuDNN 9.5 for CUDA 12.x")
        print("3. Extract and copy files to CUDA installation directory")

    return cudnn_found


def check_environment_variables():
    """Check required environment variables."""
    print("\n4. Checking environment variables...")

    issues = []

    # Check CUDA_PATH
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and os.path.exists(cuda_path):
        print(f"✓ CUDA_PATH is set: {cuda_path}")
    else:
        issues.append("CUDA_PATH not set or invalid")

    # Check if CUDA bin is in PATH
    path_var = os.environ.get("PATH", "")
    cuda_in_path = any("CUDA" in p and "bin" in p for p in path_var.split(";"))
    if cuda_in_path:
        print("✓ CUDA bin directory is in PATH")
    else:
        issues.append("CUDA bin directory not in PATH")

    if issues:
        print("\n✗ Environment variable issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("1. Add CUDA_PATH environment variable pointing to CUDA installation")
        print("2. Add %CUDA_PATH%\\bin to PATH environment variable")
        print("3. Restart your terminal/IDE after making changes")

    return len(issues) == 0


def check_tensorflow_gpu():
    """Check TensorFlow GPU configuration."""
    print("\n5. Checking TensorFlow GPU support...")

    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

        # Try to list GPUs
        gpus = tf.config.list_physical_devices("GPU")
        print(f"GPUs detected: {len(gpus)}")

        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")

            # Test GPU computation
            print("\nTesting GPU computation...")
            with tf.device("/GPU:0"):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✓ GPU computation successful: {c.numpy()}")
        else:
            print("✗ No GPUs detected by TensorFlow")

    except Exception as e:
        print(f"✗ Error testing TensorFlow GPU: {e}")


def main():
    """Run all checks."""
    print("=" * 60)
    print("CUDA and GPU Setup Checker for TensorFlow")
    print("=" * 60)

    # Check each component
    has_driver = check_nvidia_smi()
    has_cuda, cuda_version = check_cuda_paths()
    has_cudnn = check_cudnn()
    has_env = check_environment_variables()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_good = has_driver and has_cuda and has_cudnn and has_env

    if all_good:
        print("✓ All components seem to be installed")
        print("\nNow testing TensorFlow GPU support...")
        check_tensorflow_gpu()
    else:
        print("✗ Some components are missing or misconfigured")
        print("\nRequired setup for TensorFlow 2.19.0:")
        print("1. NVIDIA GPU driver (latest)")
        print("2. CUDA 12.5")
        print("3. cuDNN 9.5")
        print("4. Proper environment variables")
        print("\nPlease install missing components and restart your terminal.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
