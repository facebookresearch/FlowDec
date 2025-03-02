/* 
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# We copied the code from
# https://github.com/yang-song/score_sde_pytorch/tree/main
# The original copyright is as follows
#
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include <torch/extension.h>


torch::Tensor upfirdn2d_op(const torch::Tensor& input, const torch::Tensor& kernel,
                            int up_x, int up_y, int down_x, int down_y,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor upfirdn2d(const torch::Tensor& input, const torch::Tensor& kernel,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1) {
    CHECK_CUDA(input);
    CHECK_CUDA(kernel);

    return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
}