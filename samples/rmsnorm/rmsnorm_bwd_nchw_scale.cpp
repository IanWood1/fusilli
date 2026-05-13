// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "rmsnorm_utils.h"
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

TEST_CASE("RMS normalization backward; NCHW layout; with scale",
          "[rmsnorm][graph]") {
  constexpr int64_t n = 2, c = 3, h = 4, w = 4;
  constexpr float scale = 1.25f;
  constexpr float eps = 1e-5f;

  auto buildNewGraph = [=](const Handle &handle) {
    auto graph = std::make_shared<Graph>();
    graph->setName("rmsnorm_bwd_sample_nchw_scale");
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
    auto invRmsT = graph->tensor(TensorAttr()
                                     .setName("inv_rms")
                                     .setDim({n, 1, 1, 1})
                                     .setStride({1, 1, 1, 1}));

    auto rmsnormBwdAttr = RmsnormBwdAttr().setName("rmsnorm_bwd");

    auto [dxT, dscaleT] =
        graph->rmsnormBwd(dyT, xT, scaleT, invRmsT, rmsnormBwdAttr);

    dxT->setName("dx").setDataType(DataType::Float).setOutput(true);
    dscaleT->setName("dscale").setDataType(DataType::Float).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    return std::make_tuple(graph, dyT, xT, scaleT, invRmsT, dxT, dscaleT);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto [graph, dyT, xT, scaleT, invRmsT, dxT, dscaleT] = buildNewGraph(handle);

  auto [xVals, scaleVals, invRmsVals, dyVals, expectedDX, expectedDScale] =
      rmsnorm_utils::generateIOTensorsForBackward(n, c, h, w, scale, eps);

  FUSILLI_REQUIRE_ASSIGN(auto dyBuf, allocateBufferOfType(handle, dyT, dyVals));
  FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, xVals));
  FUSILLI_REQUIRE_ASSIGN(auto scaleBuf,
                         allocateBufferOfType(handle, scaleT, scaleVals));
  FUSILLI_REQUIRE_ASSIGN(auto invRmsBuf,
                         allocateBufferOfType(handle, invRmsT, invRmsVals));
  FUSILLI_REQUIRE_ASSIGN(
      auto dxBuf, allocateBufferOfType(handle, dxT, DataType::Float, 0.0f));
  FUSILLI_REQUIRE_ASSIGN(
      auto dscaleBuf,
      allocateBufferOfType(handle, dscaleT, DataType::Float, 0.0f));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},         {xT, xBuf},   {scaleT, scaleBuf},
          {invRmsT, invRmsBuf}, {dxT, dxBuf}, {dscaleT, dscaleBuf},
      };

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));
  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<float> dxVals;
  std::vector<float> dscaleVals;
  FUSILLI_REQUIRE_OK(dxBuf->read(handle, dxVals));
  FUSILLI_REQUIRE_OK(dscaleBuf->read(handle, dscaleVals));

  REQUIRE(dxVals.size() == expectedDX.size());
  REQUIRE(dscaleVals.size() == expectedDScale.size());
  constexpr float tolerance = 1e-4f;
  for (size_t i = 0; i < dxVals.size(); ++i)
    REQUIRE(std::abs(dxVals[i] - expectedDX[i]) < tolerance);
  for (size_t i = 0; i < dscaleVals.size(); ++i)
    REQUIRE(std::abs(dscaleVals[i] - expectedDScale[i]) < tolerance);
}
