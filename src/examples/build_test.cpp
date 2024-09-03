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

#include <iostream>
#include <cppflow/cppflow.h>

int main()
{
    // Create a tensor from a list, a = [1.0, 2.0, 3.0]
    auto a = cppflow::tensor({1.0, 2.0, 3.0});
    // Create a tensor of shape 3 filled with 1.0, b = [1.0, 1.0, 1.0]
    auto b = cppflow::fill({3}, 1.0);

    std::cout << a + b << std::endl;

    return 0;
}