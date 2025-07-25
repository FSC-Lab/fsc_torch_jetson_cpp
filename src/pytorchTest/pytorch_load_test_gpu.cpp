/*************************************************************

Copyright (C) 2024  Longhao Qian

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*************************************************************/

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <filesystem>
#include <Eigen/Dense>

void get_model_output(torch::jit::script::Module &module, torch::DeviceType device_type,Eigen::Ref<const Eigen::MatrixXd> input, Eigen::Ref<Eigen::MatrixXd> output) {

    // Create tensor directly in float32 to avoid double conversion
    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat);
    if (device_type == torch::kCUDA) {
        tensor_options = tensor_options.device(torch::Device(torch::kCUDA, 0));
    }
    
    torch::Tensor input_tensor = torch::zeros({1, input.rows()}, tensor_options);
    // Copy data directly without clone() to avoid unnecessary allocation
    auto cpu_tensor = torch::from_blob(const_cast<double*>(input.data()), {1, input.rows()}, torch::kDouble);
    input_tensor.copy_(cpu_tensor.to(torch::kFloat));

    std::vector<torch::jit::IValue> model_input;
    model_input.push_back(input_tensor);

    torch::jit::IValue model_output = module.forward(model_input);

    // Convert encoder output to tensor and copy to CPU if needed
    torch::Tensor output_tensor = model_output.toTensor();
    if (device_type == torch::kCUDA) {
        output_tensor = output_tensor.to(torch::kCPU);
    }

    // Convert back to double precision for Eigen compatibility
    output_tensor = output_tensor.to(torch::kDouble);

    // Remove batch dimension and ensure contiguous memory layout
    output_tensor = output_tensor.squeeze(0).contiguous();
    // Copy encoder output data to Eigen matrix latent_state
    std::memcpy(output.data(), output_tensor.data_ptr<double>(), output.rows() * sizeof(double));

}

int main(int argc, const char *argv[])
{

        // if (argc != 2) {
    //     std::cout << "usage: example-app <path-to-exported-script-module>\n";
    //     return -1;
    // }

    torch::DeviceType device_type;
    std::string model_path;
    if (torch::cuda::is_available()) {
        std::cout << "cuda is avaliable!\n";
        device_type = torch::kCUDA;
        model_path = std::filesystem::current_path().string() + "/python/array_net_model_cuda.pt";
    } else {
        device_type = torch::kCPU;
        model_path = std::filesystem::current_path().string() + "/python/array_net_model_cpu.pt";
    }

    std::cout << device_type << '\n';
    torch::Device device(device_type, 0);
    
    torch::jit::script::Module module_;

    std::cout << "loading array_net model\n";
    try {
        module_ = torch::jit::load(model_path);
    } catch (const c10::Error &e) {
        std::cout << "error loading the array_net model\n";
        return -1;
    }

    // load the device and set it to eval mode
    module_.to(device);
    module_.eval();

    std::cout << "ok\n";
    // Create a vector of inputs.

    Eigen::Matrix<double, 5, 1> input;
    Eigen::Matrix<double, 5, 1> output;
    input.setRandom();
    output.setZero();


    // Execute the model and turn its output into a tensor.
    uint32_t numOfCalls = 1;
    auto tic = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < numOfCalls; i++) {
        try {
            get_model_output(module_, device_type, input, output);
        } catch (const c10::Error &e) {
            std::cout << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    auto toc = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    std::cout << static_cast<double>(loop_time) / (numOfCalls * 1e6 * 1.0) << "ms \n";
    
    std::cout << "model input\n";
    std::cout << input << '\n';
    std::cout << "model output\n";
    std::cout << output << '\n';
}
