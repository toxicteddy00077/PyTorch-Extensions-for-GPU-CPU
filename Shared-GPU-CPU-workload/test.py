import torch
import gpu_conv2d_extension  # This is the name defined in setup.py

def main():
    # Create an input tensor (5x5) and a filter tensor (3x3) filled with ones.
    input_tensor = torch.rand((5, 5), dtype=torch.float32)
    filter_tensor = torch.rand((3, 3), dtype=torch.float32)
    
    # Call the custom conv2d function from your extension.
    output = gpu_conv2d_extension.my_xpu_conv2d(input_tensor, filter_tensor)
    
    # Print the output tensor.
    print("Convolution output:")
    print(output)

if __name__ == "__main__":
    main()
