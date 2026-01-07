// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace fusilli;

// This sample demonstrates channel reduction (reducing C dimension)
// Input: [N, C, H, W] = [1, 8, 2, 2]
// Output: [N, H, W] = [1, 2, 2] (keeping spatial, reducing channels)

template <typename T>
static ErrorObject runReduction(ReductionAttr::Mode mode,
                                const std::string &modeName) {
  int64_t n = 1, c = 8, h = 2, w = 2;

  auto graph = std::make_shared<Graph>();
  graph->setName("reduction_" + modeName + "_channel");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // Create input tensor [1, 8, 2, 2]
  auto xT = graph->tensor(TensorAttr()
                              .setName("x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1}));

  // Create reduction operation
  auto reductionAttr =
      ReductionAttr().setMode(mode).setName(modeName + "_channel");

  auto yT = graph->reduction(xT, reductionAttr);

  // Set output to [1, 1, 2, 2] - reducing channel dimension but keeping it as 1
  yT->setDim({n, 1, h, w}).setStride({h * w, h * w, w, 1});
  yT->setName("result").setOutput(true);

  // Validate graph
  FUSILLI_CHECK_ERROR(graph->validate());

  // Create handle for CPU backend
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));

  // Compile graph
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate and initialize input buffer
  int64_t inputSize = n * c * h * w;
  std::vector<T> inputData(inputSize);
  for (int64_t i = 0; i < inputSize; ++i) {
    inputData[i] = static_cast<T>((i * 7) % 100 - 50); // Values -50 to 49
  }

  auto xBuf = std::make_shared<Buffer>(FUSILLI_TRY(
      Buffer::allocate(handle, castToSizeT(xT->getPhysicalDim()), inputData)));

  // Allocate output buffer
  auto yBuf = std::make_shared<Buffer>(
      FUSILLI_TRY(Buffer::allocate(handle, castToSizeT(yT->getPhysicalDim()),
                                   std::vector<T>(yT->getVolume(), T(0)))));

  // Execute graph
  FUSILLI_CHECK_ERROR(graph->execute(handle, {{xT, xBuf}, {yT, yBuf}}));

  // Read output
  std::vector<T> result;
  FUSILLI_CHECK_ERROR(yBuf->read(handle, result));

  // Compute expected output
  std::vector<T> expected(h * w);
  for (int64_t hw = 0; hw < h * w; ++hw) {
    T value = (mode == ReductionAttr::Mode::MIN)
                  ? std::numeric_limits<T>::max()
                  : std::numeric_limits<T>::lowest();

    for (int64_t ch = 0; ch < c; ++ch) {
      int64_t idx = ch * h * w + hw;
      if (mode == ReductionAttr::Mode::MIN) {
        value = std::min(value, inputData[idx]);
      } else {
        value = std::max(value, inputData[idx]);
      }
    }
    expected[hw] = value;
  }

  // Verify correctness
  for (size_t i = 0; i < result.size(); ++i) {
    REQUIRE(std::abs(result[i] - expected[i]) < T(0.01));
  }
  return ok();
}

TEST_CASE("Reduction MIN channel", "[reduction][sample]") {
  FUSILLI_REQUIRE_OK(runReduction<float>(ReductionAttr::Mode::MIN, "min"));
}

TEST_CASE("Reduction MAX channel", "[reduction][sample]") {
  FUSILLI_REQUIRE_OK(runReduction<float>(ReductionAttr::Mode::MAX, "max"));
}
