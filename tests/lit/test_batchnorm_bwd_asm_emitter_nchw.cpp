// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(
// TORCH-CHECK:       %batchnorm_bwd_DY_batchnorm_bwd_perm = torch.aten.permute %batchnorm_bwd_DY
// TORCH-CHECK:       %batchnorm_bwd_X_batchnorm_bwd_perm = torch.aten.permute %batchnorm_bwd_X
// TORCH-CHECK:       %scale_bc_batchnorm_bwd = torch.aten.view %batchnorm_bwd_SCALE, %broadcast_shape_batchnorm_bwd : !torch.vtensor<[16],f32>, !torch.list<int> -> !torch.vtensor<[1,16,1,1],f32>
// TORCH-CHECK:       %mean_bc_batchnorm_bwd = torch.aten.view %batchnorm_bwd_MEAN, %broadcast_shape_batchnorm_bwd : !torch.vtensor<[16],f32>, !torch.list<int> -> !torch.vtensor<[1,16,1,1],f32>
// TORCH-CHECK:       %inv_variance_bc_batchnorm_bwd = torch.aten.view %batchnorm_bwd_INV_VARIANCE, %broadcast_shape_batchnorm_bwd : !torch.vtensor<[16],f32>, !torch.list<int> -> !torch.vtensor<[1,16,1,1],f32>
// TORCH-CHECK:       %x_hat_batchnorm_bwd = torch.aten.mul.Tensor
// TORCH-CHECK:       %batchnorm_bwd_DBIAS_batchnorm_bwd_perm = torch.aten.sum.dim_IntList %batchnorm_bwd_DY_batchnorm_bwd_perm
// TORCH-CHECK:       %batchnorm_bwd_DSCALE_batchnorm_bwd_perm = torch.aten.sum.dim_IntList %dy_x_hat_batchnorm_bwd
// TORCH-CHECK:       %mean_dy_batchnorm_bwd = torch.aten.div.Scalar %dbias_bc_batchnorm_bwd
// TORCH-CHECK:       %mean_dy_xhat_batchnorm_bwd = torch.aten.div.Scalar %dscale_bc_batchnorm_bwd
// TORCH-CHECK:       %batchnorm_bwd_DX_batchnorm_bwd_perm = torch.aten.mul.Tensor %dx_inner_batchnorm_bwd, %scale_inv_variance_batchnorm_bwd
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_bwd_DBIAS overwrites %batchnorm_bwd_DBIAS_
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_bwd_DSCALE overwrites %batchnorm_bwd_DSCALE_
// TORCH-CHECK:       torch.overwrite.tensor.contents %batchnorm_bwd_DX overwrites %batchnorm_bwd_DX_
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// STATS-CHECK: "dispatch-count":
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testBatchnormBwdAsmEmitterNchw(const std::string &mode) {
  int64_t n = 4, c = 16, h = 8, w = 8;
  auto graph = std::make_shared<Graph>();
  graph->setName("batchnorm_bwd_asm_emitter_nchw");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto dyT = graph->tensor(TensorAttr()
                               .setName("batchnorm_bwd_DY")
                               .setDim({n, c, h, w})
                               .setStride({c * h * w, h * w, w, 1}));
  auto xT = graph->tensor(TensorAttr()
                              .setName("batchnorm_bwd_X")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1}));
  auto scaleT = graph->tensor(
      TensorAttr().setName("batchnorm_bwd_SCALE").setDim({c}).setStride({1}));
  auto meanT = graph->tensor(
      TensorAttr().setName("batchnorm_bwd_MEAN").setDim({c}).setStride({1}));
  auto invVarianceT = graph->tensor(TensorAttr()
                                        .setName("batchnorm_bwd_INV_VARIANCE")
                                        .setDim({c})
                                        .setStride({1}));

  auto batchnormBwdAttr = BatchnormBwdAttr().setName("batchnorm_bwd");
  auto [dxT, dscaleT, dbiasT] = graph->batchnormBwd(
      dyT, xT, scaleT, meanT, invVarianceT, batchnormBwdAttr);

  dxT->setName("batchnorm_bwd_DX").setDataType(DataType::Float).setOutput(true);
  dscaleT->setName("batchnorm_bwd_DSCALE")
      .setDataType(DataType::Float)
      .setOutput(true);
  dbiasT->setName("batchnorm_bwd_DBIAS")
      .setDataType(DataType::Float)
      .setOutput(true);

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

  auto status = testBatchnormBwdAsmEmitterNchw(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
