// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "rope_utils.h"
#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// RoPE via custom op. Tests identity and 90-degree rotation.
// Parameterized over dtype and shape.
TEST_CASE("Custom op: RoPE (Rotary Position Embedding)", "[custom_op][rope]") {
  auto execute = [&]<typename T>(Handle &handle, DataType dt, int64_t batch,
                                 int64_t seqLen, int64_t heads, int64_t headDim,
                                 const std::vector<T> &xData,
                                 const std::vector<T> &cosData,
                                 const std::vector<T> &sinData,
                                 const std::vector<T> &expected) {
    auto graph = std::make_shared<Graph>();
    graph->setName(std::format("rope_dt{}_B{}S{}H{}D{}",
                               kDataTypeToMlirTypeAsm.at(dt), batch, seqLen,
                               heads, headDim));
    graph->setIODataType(dt).setIntermediateDataType(dt);

    std::vector<int64_t> xDim = {batch, seqLen, heads, headDim};
    std::vector<int64_t> csDim = {1, seqLen, 1, headDim / 2};

    auto xStride =
        generateStrideFromDim(xDim, getContiguousStrideOrder(xDim.size()));
    auto csStride =
        generateStrideFromDim(csDim, getContiguousStrideOrder(csDim.size()));

    auto xT = graph->tensor(
        TensorAttr().setName("x").setDim(xDim).setStride(xStride));
    auto cosT = graph->tensor(
        TensorAttr().setName("cos").setDim(csDim).setStride(csStride));
    auto sinT = graph->tensor(
        TensorAttr().setName("sin").setDim(csDim).setStride(csStride));

    CustomOpAttr ropeAttr;
    ropeAttr.setName("rope").setMlir(getRopeMlir()).setNumOutputs(1);

    auto outs = graph->customOp({xT, cosT, sinT}, ropeAttr);
    outs[0]->setDim(xDim).setStride(xStride).setDataType(dt).setOutput(true);

    FUSILLI_REQUIRE_OK(graph->validate());
    FUSILLI_REQUIRE_OK(graph->compile(handle, /*remove=*/true));

    FUSILLI_REQUIRE_ASSIGN(auto xBuf, allocateBufferOfType(handle, xT, xData));
    FUSILLI_REQUIRE_ASSIGN(auto cosBuf,
                           allocateBufferOfType(handle, cosT, cosData));
    FUSILLI_REQUIRE_ASSIGN(auto sinBuf,
                           allocateBufferOfType(handle, sinT, sinData));
    FUSILLI_REQUIRE_ASSIGN(auto outBuf,
                           allocateBufferOfType(handle, outs[0], dt, 0.0));

    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>>
        variantPack = {
            {xT, xBuf}, {cosT, cosBuf}, {sinT, sinBuf}, {outs[0], outBuf}};

    FUSILLI_REQUIRE_ASSIGN(
        auto workspace, allocateWorkspace(handle, graph->getWorkspaceSize()));

    FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

    std::vector<T> result;
    FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
      REQUIRE(result[i] == expected[i]);
  };

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Shape: B=1, S=1, H=1, D=4
  // cos/sin shape: [1, 1, 1, 2]

  SECTION("Float32: identity rotation (cos=1, sin=0)") {
    // x = {1,2,3,4}, cos=1, sin=0 → output = x
    execute(handle, DataType::Float,
            /*batch=*/1, /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
            /*xData=*/std::vector<float>{1, 2, 3, 4},
            /*cosData=*/std::vector<float>{1, 1},
            /*sinData=*/std::vector<float>{0, 0},
            /*expected=*/std::vector<float>{1, 2, 3, 4});
  }

  SECTION("Float32: 90-degree rotation (cos=0, sin=1)") {
    // x = {1,2,3,4} → x1={1,2}, x2={3,4}
    // out1 = x1*0 - x2*1 = {-3,-4}
    // out2 = x2*0 + x1*1 = {1,2}
    // result = cat({-3,-4}, {1,2}) = {-3,-4,1,2}
    execute(handle, DataType::Float,
            /*batch=*/1, /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
            /*xData=*/std::vector<float>{1, 2, 3, 4},
            /*cosData=*/std::vector<float>{0, 0},
            /*sinData=*/std::vector<float>{1, 1},
            /*expected=*/std::vector<float>{-3, -4, 1, 2});
  }

  SECTION("Float16: identity rotation") {
    execute(handle, DataType::Half,
            /*batch=*/1, /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
            /*xData=*/std::vector<half>{half(1), half(2), half(3), half(4)},
            /*cosData=*/std::vector<half>{half(1), half(1)},
            /*sinData=*/std::vector<half>{half(0), half(0)},
            /*expected=*/std::vector<half>{half(1), half(2), half(3), half(4)});
  }

  SECTION("Float16: 90-degree rotation") {
    execute(handle, DataType::Half,
            /*batch=*/1, /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
            /*xData=*/std::vector<half>{half(1), half(2), half(3), half(4)},
            /*cosData=*/std::vector<half>{half(0), half(0)},
            /*sinData=*/std::vector<half>{half(1), half(1)},
            /*expected=*/
            std::vector<half>{half(-3), half(-4), half(1), half(2)});
  }
}
