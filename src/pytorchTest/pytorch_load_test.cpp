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

int main(int argc, const char *argv[])
{
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../../../python/array_net_model.pt");
    } catch (const c10::Error &e) {
        std::cout << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor({1.0, 1.0, 1.0, 1.0, 1.0}));
    // at::Tensor output;
    torch::jit::IValue output;
    // Execute the model and turn its output into a tensor.
    auto tic = std::chrono::steady_clock::now();
    uint32_t numOfCalls = 1000;
    for (uint32_t i = 0; i < numOfCalls; i++) {
        output = module.forward(inputs).toTensor();
    }
    auto toc = std::chrono::steady_clock::now();
    auto loop_time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic).count();
    std::cout << static_cast<double>(loop_time) / (numOfCalls * 1e6 * 1.0)<< "ms \n";
    std::cout << output << '\n';
}