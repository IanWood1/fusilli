// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

// This sample demonstrates spatial reduction (reducing H and W dimensions)
// Input: [N, C, H, W] = [1, 3, 4, 4]
// Output: [N, C, 1, 1] = [1, 3, 1, 1]

TEST_CASE("Reduction sum spatial", "[reduction][sample]") {
  int64_t n = 1, c = 3, h = 4, w = 4;

  auto graph = std::make_shared<Graph>();
  graph->setName("reduction_sum_spatial");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // Create input tensor [1, 3, 4, 4]
  auto xT = graph->tensor(TensorAttr()
                              .setName("x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1}));

  // Create reduction operation
  auto reductionAttr =
      ReductionAttr().setMode(ReductionAttr::Mode::SUM).setName("spatial_sum");

  auto yT = graph->reduction(xT, reductionAttr);

  // Set output to [1, 3, 1, 1] - reducing spatial dimensions
  yT->setDim({n, c, 1, 1}).setStride({c, 1, 1, 1});
  yT->setName("result").setOutput(true);

  // Validate graph
  FUSILLI_REQUIRE_OK(graph->validate());

  // Create handle for CPU backend
  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU));

  // Compile graph
  FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

  // Allocate and initialize input buffer with sequential values
  int64_t inputSize = n * c * h * w;
  std::vector<float> inputData(inputSize);
  for (int64_t i = 0; i < inputSize; ++i) {
    inputData[i] = static_cast<float>(i + 1);
  }

  auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT(xT->getPhysicalDim()), inputData)));

  // Allocate output buffer
  auto yBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
      Buffer::allocate(handle, castToSizeT(yT->getPhysicalDim()),
                       std::vector<float>(yT->getVolume(), 0.0f))));

  // Execute graph
  FUSILLI_REQUIRE_OK(graph->execute(handle, {{xT, xBuf}, {yT, yBuf}}));

  // Read output
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(yBuf->read(handle, result));

  // Verify correctness
  for (int64_t i = 0; i < n * c; ++i) {
    float expected = 0.0f;
    for (int64_t j = 0; j < h * w; ++j) {
      expected += inputData[i * h * w + j];
    }
    REQUIRE(std::abs(result[i] - expected) < 0.01f);
  }
}
