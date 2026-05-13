// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main
// TORCH-CHECK:       %layernorm_bwd_EPSILON = torch.vtensor.literal(dense<0x3727C5AC> : tensor<1xf32>) : !torch.vtensor<[1],f32>
// TORCH-CHECK:       %normalized_dims_layernorm_bwd = torch.prim.ListConstruct %normalized_dims_val_0_layernorm_bwd, %normalized_dims_val_1_layernorm_bwd, %normalized_dims_val_2_layernorm_bwd : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %parameter_dims_layernorm_bwd = torch.prim.ListConstruct %parameter_dims_val_0_layernorm_bwd : (!torch.int) -> !torch.list<int>
// TORCH-CHECK:       %input_shape_layernorm_bwd = torch.prim.ListConstruct %input_shape_val_0_layernorm_bwd, %input_shape_val_1_layernorm_bwd, %input_shape_val_2_layernorm_bwd, %input_shape_val_3_layernorm_bwd : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %mean_expanded_layernorm_bwd = torch.aten.expand %mean_layernorm_bwd_perm, %input_shape_layernorm_bwd, %expand_implicit_layernorm_bwd : !torch.vtensor<[2,1,1,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3,4,5],f32>
// TORCH-CHECK:       %input_zero_mean_layernorm_bwd = torch.aten.sub.Tensor %x_layernorm_bwd_perm, %mean_expanded_layernorm_bwd, %alpha_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,4,5],f32>, !torch.int -> !torch.vtensor<[2,3,4,5],f32>
// TORCH-CHECK:       %input_normalized_layernorm_bwd = torch.aten.mul.Tensor %input_zero_mean_layernorm_bwd, %inv_variance_expanded_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,4,5],f32> -> !torch.vtensor<[2,3,4,5],f32>
// TORCH-CHECK:       %grad_out_weighted_layernorm_bwd = torch.aten.mul.Tensor %dy_layernorm_bwd_perm, %scale_layernorm_bwd_perm : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[1,3,4,5],f32> -> !torch.vtensor<[2,3,4,5],f32>
// TORCH-CHECK:       %grad_out_weighted_mean_layernorm_bwd = torch.aten.mean.dim %grad_out_weighted_layernorm_bwd, %normalized_dims_layernorm_bwd, %true_layernorm_bwd, %none_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,1,1,1],f32>
// TORCH-CHECK:       %grad_out_weighted_input_normalized_mean_layernorm_bwd = torch.aten.mean.dim %grad_out_weighted_input_normalized_layernorm_bwd, %normalized_dims_layernorm_bwd, %true_layernorm_bwd, %none_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,1,1,1],f32>
// TORCH-CHECK:       %dx_layernorm_bwd_perm = torch.aten.mul.Tensor %centered_grad_layernorm_bwd, %inv_variance_expanded_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.vtensor<[2,3,4,5],f32> -> !torch.vtensor<[2,3,4,5],f32>
// TORCH-CHECK:       %dscale_layernorm_bwd_perm = torch.aten.sum.dim_IntList %grad_weight_input_layernorm_bwd, %parameter_dims_layernorm_bwd, %keep_param_dims_layernorm_bwd, %none_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,3,4,5],f32>
// TORCH-CHECK:       %dbias_layernorm_bwd_perm = torch.aten.sum.dim_IntList %dy_layernorm_bwd_perm, %parameter_dims_layernorm_bwd, %keep_param_dims_layernorm_bwd, %none_layernorm_bwd : !torch.vtensor<[2,3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,3,4,5],f32>
// TORCH-CHECK-NOT:   torch.aten.native_layer_norm_backward
// TORCH-CHECK:       torch.overwrite.tensor.contents %dbias overwrites %dbias_
// TORCH-CHECK:       torch.overwrite.tensor.contents %dscale overwrites %dscale_
// TORCH-CHECK:       torch.overwrite.tensor.contents %dx overwrites %dx_
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testLayernormBwdAsmEmitterNchw() {
  constexpr int64_t n = 2, c = 3, h = 4, w = 5;
  auto graph = std::make_shared<Graph>();
  graph->setName("layernorm_bwd_asm_emitter_nchw");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto dyT = graph->tensor(TensorAttr()
                               .setName("dy")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, h * w, w, 1}));
  auto xT = graph->tensor(TensorAttr()
                              .setName("x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1}));
  auto scaleT = graph->tensor(TensorAttr()
                                  .setName("scale")
                                  .setDim({1, c, h, w})
                                  .setStride({c * h * w, h * w, w, 1}));
  auto meanT = graph->tensor(TensorAttr()
                                 .setName("mean")
                                 .setDim({n, 1, 1, 1})
                                 .setStride({1, 1, 1, 1}));
  auto invVarianceT = graph->tensor(TensorAttr()
                                        .setName("inv_variance")
                                        .setDim({n, 1, 1, 1})
                                        .setStride({1, 1, 1, 1}));
  auto epsilonT = graph->tensor(TensorAttr(1e-5f));

  auto layernormBwdAttr =
      LayernormBwdAttr().setEpsilon(epsilonT).setName("layernorm_bwd");
  auto [dxT, dscaleT, dbiasT] = graph->layernormBwd(
      dyT, xT, scaleT, meanT, invVarianceT, layernormBwdAttr);

  dxT->setName("dx").setDataType(DataType::Float).setOutput(true);
  dscaleT->setName("dscale").setDataType(DataType::Float).setOutput(true);
  dbiasT->setName("dbias").setDataType(DataType::Float).setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());
  FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
  FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
  std::cout << generatedAsm << std::endl;

  return ok();
}

int main() {
  auto status = testLayernormBwdAsmEmitterNchw();
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
