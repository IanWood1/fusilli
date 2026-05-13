// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "batchnorm_utils.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

TEST_CASE("Batch normalization backward; NCHW layout; scale",
          "[batchnorm_bwd][graph]") {
  constexpr int64_t n = 2, c = 4, h = 4, w = 4;
  constexpr float scale = 0.5f;
  constexpr float eps = 0.5f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("batchnorm_bwd_sample_nchw_scale");
    graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

    auto dyT = graph->tensor(TensorAttr()
                                 .setName("dy")
                                 .setDim({n, c, h, w})
                                 .setStride({c * h * w, h * w, w, 1}));
    auto xT = graph->tensor(TensorAttr()
                                .setName("x")
                                .setDim({n, c, h, w})
                                .setStride({c * h * w, h * w, w, 1}));

    auto scaleT = graph->tensor(TensorAttr().setName("scale"));
    auto meanT = graph->tensor(TensorAttr().setName("saved_mean"));
    auto invVarianceT =
        graph->tensor(TensorAttr().setName("saved_inv_variance"));

    auto batchnormBwdAttr = BatchnormBwdAttr().setName("batchnorm_bwd");
    auto [dxT, dscaleT, dbiasT] = graph->batchnormBwd(
        dyT, xT, scaleT, meanT, invVarianceT, batchnormBwdAttr);

    dxT->setName("dx").setDataType(DataType::Float).setOutput(true);
    dscaleT->setName("dscale").setDataType(DataType::Float).setOutput(true);
    dbiasT->setName("dbias").setDataType(DataType::Float).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, scaleT, meanT, invVarianceT, dxT,
                           dscaleT, dbiasT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, xT, scaleT, meanT, invVarianceT, dxT, dscaleT, dbiasT] =
      buildNewGraph(handle);

  auto [dyVals, xVals, savedMeanVals, savedInvVarVals, expectedDxVals,
        expectedDscaleVals, expectedDbiasVals] =
      batchnorm_utils::generateNchwForBackward(n, c, h, w, scale, eps);

  FUSILLI_REQUIRE_ASSIGN(auto dyBuf, allocateBufferOfType(handle, dyT, dyVals));
  FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, xVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto scaleBuf,
      allocateBufferOfType(handle, scaleT, DataType::Float, scale));
  FUSILLI_REQUIRE_ASSIGN(auto meanBuf,
                         allocateBufferOfType(handle, meanT, savedMeanVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto invVarianceBuf,
      allocateBufferOfType(handle, invVarianceT, savedInvVarVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto dxBuf, allocateBufferOfType(handle, dxT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto dscaleBuf,
      allocateBufferOfType(handle, dscaleT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto dbiasBuf,
      allocateBufferOfType(handle, dbiasT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {scaleT, scaleBuf},
          {meanT, meanBuf},
          {invVarianceT, invVarianceBuf},
          {dxT, dxBuf},
          {dscaleT, dscaleBuf},
          {dbiasT, dbiasBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> dxVals, dscaleVals, dbiasVals;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));
  FUSILLI_REQUIRE_OK(dscaleBuf->read(handle, dscaleVals));
  FUSILLI_REQUIRE_OK(dbiasBuf->read(handle, dbiasVals));

  constexpr float tolerance = 1e-4f;
  REQUIRE(dxVals.size() == expectedDxVals.size());
  REQUIRE(dscaleVals.size() == expectedDscaleVals.size());
  REQUIRE(dbiasVals.size() == expectedDbiasVals.size());

  for (size_t i = 0; i < dxVals.size(); ++i)
    REQUIRE(std::abs(dxVals[i] - expectedDxVals[i]) < tolerance);
  for (size_t i = 0; i < dscaleVals.size(); ++i) {
    REQUIRE(std::abs(dscaleVals[i] - expectedDscaleVals[i]) < tolerance);
    REQUIRE(std::abs(dbiasVals[i] - expectedDbiasVals[i]) < tolerance);
  }
}
