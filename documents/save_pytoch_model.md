# PyTorch Model Deployment in C++

This guide explains how to export PyTorch models for use in C++ applications using TorchScript.

## Overview

To use a PyTorch model in C++, you need to:
1. Convert your `nn.Module` to TorchScript format in Python
2. Save the model as a `.pt` file
3. Load and use the model in C++ with LibTorch

## Step 1: Export PyTorch Model from Python

### Prerequisites
- Your model must inherit from `nn.Module`
- The model output comes from the `forward()` method
- Choose between tracing or scripting based on your model complexity

### Option A: Tracing (`torch.jit.trace`)
**Best for**: Models with consistent control flow (no data-dependent conditionals or loops)

```python
import torch
import torch.nn as nn

# Example model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Create dummy input with correct shape
dummy_input = torch.randn(1, 2)  # Fixed: input size matches model

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)

# Save the traced model
torch.jit.save(traced_model, "traced_model.pt")
```

### Option B: Scripting (`torch.jit.script`)
**Best for**: Models with control flow (if statements, loops, etc.)

```python
import torch
import torch.nn as nn

class MyScriptedModel(nn.Module):
    def __init__(self):
        super(MyScriptedModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        if x.mean() > 0:
            return self.linear(x) * 2
        else:
            return self.linear(x)

# Create and script the model
model = MyScriptedModel()
scripted_model = torch.jit.script(model)

# Save the scripted model
torch.jit.save(scripted_model, "scripted_model.pt")
```

### Important Considerations

#### Set Model to Evaluation Mode
For models with batch normalization or dropout:

```python
model.eval()  # Set to evaluation mode
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, "scripted_model.pt")
```

#### Save Both CPU and CUDA Versions
It's recommended to save device-specific versions for optimal performance:

```python
# Convert CUDA model to CPU
cuda_model = torch.jit.load("cuda_model.pt", map_location="cpu")
torch.jit.save(cuda_model, "cpu_model.pt")

# Convert CPU model to CUDA
cpu_model = torch.jit.load("cpu_model.pt", map_location="cuda")
torch.jit.save(cpu_model, "cuda_model.pt")
```

## Step 2: Load PyTorch Model in C++

### CMake Configuration

Add LibTorch to your `CMakeLists.txt`:

```cmake
# Find PyTorch installation path
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/${PYTORCH_PATH}/torch/share/cmake")

# Find and link LibTorch
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "LibTorch found successfully")
endif()

# Link LibTorch to your target
target_link_libraries(your_target_name ${TORCH_LIBRARIES})
```

**Find PyTorch Path:**
```bash
pip show torch
```
Look for the `Location` field in the output - this is your `PYTORCH_PATH`.

### C++ Implementation

#### Include Headers
```cpp
#include <torch/script.h>  // TorchScript module loading
#include <torch/torch.h>   // Core LibTorch functionality
#include <iostream>
#include <memory>
```

#### Device Configuration
```cpp
// Check CUDA availability
bool cuda_available = torch::cuda::is_available();
std::cout << "CUDA available: " << cuda_available << std::endl;

// Set device
torch::Device device(torch::kCPU);
if (cuda_available) {
    device = torch::Device(torch::kCUDA);
}
```

#### Load and Configure Model
```cpp
try {
    // Load the model
    torch::jit::script::Module module = torch::jit::load("your_model_path.pt");

    // Move model to device and set evaluation mode
    module.to(device);
    module.eval();

    std::cout << "Model loaded successfully" << std::endl;

} catch (const c10::Error& e) {
    std::cerr << "Error loading model: " << e.msg() << std::endl;
    return -1;
}
```

#### Model Inference

**Single Model Execution:**
```cpp
// Create input tensor
std::vector<torch::jit::IValue> inputs;
auto input_tensor = torch::randn({1, your_model_input_dim}).to(device);
inputs.push_back(input_tensor);

// Execute the model
torch::jit::IValue output = module.forward(inputs);

// Convert output to tensor if needed
torch::Tensor output_tensor = output.toTensor();
std::cout << "Output shape: " << output_tensor.sizes() << std::endl;
```

**Multiple Model Pipeline:**
```cpp
// Load multiple models
torch::jit::script::Module encoder = torch::jit::load("encoder.pt");
torch::jit::script::Module decoder = torch::jit::load("decoder.pt");

encoder.to(device).eval();
decoder.to(device).eval();

// Execute encoder
std::vector<torch::jit::IValue> encoder_inputs;
auto input_tensor = torch::randn({1, input_dim}).to(device);
encoder_inputs.push_back(input_tensor);

torch::jit::IValue encoder_output = encoder.forward(encoder_inputs);

// Execute decoder using encoder output
std::vector<torch::jit::IValue> decoder_inputs;
decoder_inputs.push_back(encoder_output);

torch::jit::IValue final_output = decoder.forward(decoder_inputs);
```

## Additional Resources

- [LibTorch C++ API Documentation](https://pytorch.org/cppdocs/)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
- [PyTorch C++ Frontend Tutorial](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
