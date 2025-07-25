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

// https://pytorch.org/tutorials/advanced/cpp_export.html

#include <torch/script.h>  // One-stop header.
#include <iostream>
#include <memory>
#include <chrono>
#include <filesystem>
#include <Eigen/Dense>

void get_model_output(torch::jit::script::Module &module,Eigen::Ref<const Eigen::MatrixXd> input, Eigen::Ref<Eigen::MatrixXd> output) {
    
    // Create a tensor from the input Eigen matrix
    torch::Tensor input_tensor = torch::from_blob(const_cast<double*>(input.data()), {1, input.rows()}, torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU)).clone();

    // Convert to float32 to match model weights
    input_tensor = input_tensor.to(torch::kFloat);

    std::vector<torch::jit::IValue> model_input;
    model_input.push_back(input_tensor);

    torch::jit::IValue model_output = module.forward(model_input);

    // Convert encoder output to tensor and copy to CPU if needed
    torch::Tensor output_tensor = model_output.toTensor();
    // Convert back to double precision for Eigen compatibility
    output_tensor = output_tensor.to(torch::kDouble);

    // Remove batch dimension and ensure contiguous memory layout
    output_tensor = output_tensor.squeeze(0).contiguous();

    // Copy encoder output data to Eigen matrix latent_state
    std::memcpy(output.data(), output_tensor.data_ptr<double>(), output.rows() * sizeof(double));

}


int main(int argc, const char *argv[])
{   
    torch::DeviceType device_type;
    device_type = torch::kCPU;
    torch::Device device(device_type, 0);

    std::string model_path = std::filesystem::current_path().string() + "/python/array_net_model_cpu.pt";
    torch::jit::script::Module module_;
    try {
        module_ = torch::jit::load(model_path);
    } catch (const c10::Error &e) {
        std::cout << "error loading the model\n";
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

    std::cout << "model input\n";
    std::cout << input << '\n';
    // Execute the model and turn its output into a tensor.
    uint32_t numOfCalls = 1000;
    auto tic = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < numOfCalls; i++) {
        try {
            get_model_output(module_, input, output);
        } catch (const c10::Error &e) {
            std::cout << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    auto toc = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    std::cout << static_cast<double>(loop_time) / (numOfCalls * 1e6 * 1.0) << "ms \n";
    std::cout << output << '\n';
}