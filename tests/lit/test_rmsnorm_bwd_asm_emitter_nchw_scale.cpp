// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK-NOT:   torch.aten.rms_norm_backward
// TORCH-CHECK:       %normalized_reduce_dims_val_0_rmsnorm_bwd = torch.constant.int 1
// TORCH-CHECK:       %normalized_reduce_dims_val_1_rmsnorm_bwd = torch.constant.int 2
// TORCH-CHECK:       %normalized_reduce_dims_val_2_rmsnorm_bwd = torch.constant.int 3
// TORCH-CHECK:       %batch_reduce_dims_val_0_rmsnorm_bwd = torch.constant.int 0
// TORCH-CHECK:       %normalized_size_rmsnorm_bwd = torch.constant.int 48
// TORCH-CHECK:       %scaled_dy_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %dy_scaled_x_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %dot_sum_rmsnorm_bwd = torch.aten.sum.dim_IntList
// TORCH-CHECK:       %mean_dot_rmsnorm_bwd = torch.aten.div.Scalar
// TORCH-CHECK:       %inv_rms_sq_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %correction_factor_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %x_correction_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %dx_base_rmsnorm_bwd = torch.aten.sub.Tensor
// TORCH-CHECK:       %dx_rmsnorm_bwd_perm = torch.aten.mul.Tensor
// TORCH-CHECK:       %dscale_full_rmsnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %dscale_rmsnorm_bwd_perm = torch.aten.sum.dim_IntList
// TORCH-CHECK-NOT:   torch.aten.rms_norm_backward
//
// AMDGPU-STATS-CHECK: "dispatch-count":
// CPU-STATS-CHECK: "dispatch-count":
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testRmsnormBwdAsmEmitterNchwScale(const std::string &mode) {
  int64_t n = 2, c = 3, h = 4, w = 4;
  auto graph = std::make_shared<Graph>();
  graph->setName("rmsnorm_bwd_asm_emitter_nchw_scale");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto dyT = graph->tensor(TensorAttr()
                               .setName("arg0_dy")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, h * w, w, 1}));
  auto xT = graph->tensor(TensorAttr()
                              .setName("arg1_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1}));
  auto scaleT = graph->tensor(TensorAttr()
                                  .setName("arg2_scale")
                                  .setDim({1, c, h, w})
                                  .setStride({c * h * w, h * w, w, 1}));
  auto invRmsT = graph->tensor(TensorAttr()
                                   .setName("arg3_inv_rms")
                                   .setDim({n, 1, 1, 1})
                                   .setStride({1, 1, 1, 1}));

  auto rmsnormBwdAttr = RmsnormBwdAttr().setName("rmsnorm_bwd");

  auto [dxT, dscaleT] =
      graph->rmsnormBwd(dyT, xT, scaleT, invRmsT, rmsnormBwdAttr);

  dxT->setName("dx").setOutput(true);
  dscaleT->setName("dscale").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    FUSILLI_ASSIGN_OR_RETURN(auto generatedAsm, graph->emitAsm());
    FUSILLI_CHECK_ERROR(checkMlirIndentation(generatedAsm));
    std::cout << generatedAsm << std::endl;
  }

  if (mode == "stats") {
    FUSILLI_ASSIGN_OR_RETURN(Handle handle, Handle::create(kDefaultBackend));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    FUSILLI_ASSIGN_OR_RETURN(auto stats, graph->readCompilationCacheFile(
                                             CachedAssetsType::Statistics));
    std::cout << stats << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testRmsnormBwdAsmEmitterNchwScale(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
