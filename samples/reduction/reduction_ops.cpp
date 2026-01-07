// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// Based on parameters, generates a unique name for the graph
static std::string generateName(ReductionAttr::Mode mode, DataType type,
                                const std::vector<int64_t> &xDim,
                                const std::vector<int64_t> &yDim) {
  std::string name =
      std::format("reduction_{}_dt{}", ReductionAttr::kModeToStr.at(mode),
                  kDataTypeToMlirTypeAsm.at(type));
  name += "_x";
  for (const auto &d : xDim) {
    name += std::format("_{}", d);
  }
  name += "_y";
  for (const auto &d : yDim) {
    name += std::format("_{}", d);
  }
  return name;
};

TEST_CASE("Reduction ops", "[reduction][graph]") {
  const auto xDims = std::vector<int64_t>{2, 16, 8, 8};
  const auto yDims = std::vector<int64_t>{2, 16, 1, 1};

  const auto mode = GENERATE(ReductionAttr::Mode::SUM, ReductionAttr::Mode::MIN,
                             ReductionAttr::Mode::MAX);

  auto execute = [&]<typename T>(const std::shared_ptr<Handle> &handlePtr,
                                 DataType dt, T initValue) {
    auto buildNewGraph = [&](const Handle &handle) {
      // Create graph
      auto graph = std::make_shared<Graph>();
      graph->setName(generateName(mode, dt, xDims, yDims));
      graph->setIODataType(dt).setComputeDataType(dt);

      // Initialize input tensor
      auto xT = graph->tensor(TensorAttr().setName("x").setDim(xDims).setStride(
          generateStrideFromDim(xDims,
                                getContiguousStrideOrder(xDims.size()))));

      // Create Reduction op
      auto reductionAttr = ReductionAttr().setMode(mode);
      auto yT = graph->reduction(xT, reductionAttr);

      // Set output dimensions for spatial reduction
      yT->setDim(yDims).setStride(
          generateStrideFromDim(yDims, getContiguousStrideOrder(yDims.size())));

      yT->setName("result").setOutput(true);

      // Validate, infer missing properties
      FUSILLI_REQUIRE_OK(graph->validate());

      // Compile
      FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

      return std::make_tuple(graph, xT, yT);
    };

    Handle &handle = *handlePtr;
    // Build graph for the given handle (device), validate and compile it.
    auto [graph, xT, yT] = buildNewGraph(handle);

    // Calculate total input size
    int64_t xSize = 1;
    for (auto d : xDims)
      xSize *= d;

    // Create input data with known pattern
    std::vector<T> xData(xSize);
    for (int64_t i = 0; i < xSize; ++i) {
      xData[i] = static_cast<T>(i % 100 - 50); // Values from -50 to 49
    }

    // Allocate input buffer with data
    auto xBuf = std::make_shared<Buffer>(FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle, castToSizeT(xT->getPhysicalDim()), xData)));

    // Allocate output buffer
    auto yBuf =
        FUSILLI_REQUIRE_UNWRAP(allocateBufferOfType(handle, yT, dt, 0.0f));

    // Create variant pack
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {xT, xBuf},
            {yT, yBuf},
        };

    // Execute graph once
    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

    // Calculate expected output
    int64_t ySize = 1;
    for (auto d : yDims)
      ySize *= d;

    std::vector<T> expectedY(ySize);

    // Compute reduction manually
    // Input is [2, 16, 8, 8], output is [2, 16]
    // We reduce over the last two dimensions (spatial)
    int64_t reducedSize = 8 * 8; // h * w
    for (int64_t n = 0; n < 2; ++n) {
      for (int64_t c = 0; c < 16; ++c) {
        int64_t outIdx = n * 16 + c;
        T result;

        switch (mode) {
        case ReductionAttr::Mode::SUM: {
          result = T(0);
          for (int64_t h = 0; h < 8; ++h) {
            for (int64_t w = 0; w < 8; ++w) {
              int64_t inIdx = ((n * 16 + c) * 8 + h) * 8 + w;
              result += xData[inIdx];
            }
          }
          break;
        }
        case ReductionAttr::Mode::MIN: {
          result = std::numeric_limits<T>::max();
          for (int64_t h = 0; h < 8; ++h) {
            for (int64_t w = 0; w < 8; ++w) {
              int64_t inIdx = ((n * 16 + c) * 8 + h) * 8 + w;
              result = std::min(result, xData[inIdx]);
            }
          }
          break;
        }
        case ReductionAttr::Mode::MAX: {
          result = std::numeric_limits<T>::lowest();
          for (int64_t h = 0; h < 8; ++h) {
            for (int64_t w = 0; w < 8; ++w) {
              int64_t inIdx = ((n * 16 + c) * 8 + h) * 8 + w;
              result = std::max(result, xData[inIdx]);
            }
          }
          break;
        }
        default:
          FAIL("Unsupported reduction mode: "
               << ReductionAttr::kModeToStr.at(mode));
        }

        expectedY[outIdx] = result;
      }
    }

    // Read output buffer
    std::vector<T> result;
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));

    // Validate output
    REQUIRE(result.size() == expectedY.size());
    for (size_t i = 0; i < result.size(); ++i) {
      // Allow small tolerance for floating point
      if constexpr (std::is_floating_point_v<T>) {
        REQUIRE(std::abs(result[i] - expectedY[i]) < T(0.01));
      } else {
        REQUIRE(result[i] == expectedY[i]);
      }
    }

    // Execute graph a few more times to ensure consistency
    constexpr size_t numIters = 3;
    for (size_t i = 0; i < numIters; i++)
      FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack));

    // Repeat output buffer checks
    result.clear();
    FUSILLI_REQUIRE_OK(yBuf->read(handle, result));
    for (size_t i = 0; i < result.size(); ++i) {
      if constexpr (std::is_floating_point_v<T>) {
        REQUIRE(std::abs(result[i] - expectedY[i]) < T(0.01));
      } else {
        REQUIRE(result[i] == expectedY[i]);
      }
    }
  };

  // Parameterize sample by backend and create device-specific handles
  std::shared_ptr<Handle> handlePtr;
  SECTION("cpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::CPU)));
  }
#ifdef FUSILLI_ENABLE_AMDGPU
  SECTION("amdgpu backend") {
    handlePtr = std::make_shared<Handle>(
        FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU)));
  }
#endif

  // int32
  execute(handlePtr, DataType::Int32, int(0));
  // fp16
  execute(handlePtr, DataType::Half, half(0.f16));
  // fp32
  execute(handlePtr, DataType::Float, float(0.0f));
}
