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

namespace {

// Build, compile, and execute a RoPE graph, returning the output.
template <typename T>
std::vector<T>
executeRope(Handle &handle, DataType dt, int64_t batch, int64_t seqLen,
            int64_t heads, int64_t headDim, const std::vector<T> &xData,
            const std::vector<T> &cosData, const std::vector<T> &sinData) {
  auto graph = std::make_shared<Graph>();
  graph->setName(std::format("test_rope_dt{}_B{}S{}H{}D{}",
                             kDataTypeToMlirTypeAsm.at(dt), batch, seqLen,
                             heads, headDim));
  graph->setIODataType(dt).setIntermediateDataType(dt);

  std::vector<int64_t> xDim = {batch, seqLen, heads, headDim};
  std::vector<int64_t> csDim = {1, seqLen, 1, headDim / 2};

  auto xT = createTestTensor("x", xDim, graph.get());
  auto cosT = createTestTensor("cos", csDim, graph.get());
  auto sinT = createTestTensor("sin", csDim, graph.get());

  CustomOpAttr ropeAttr;
  ropeAttr.setName("rope").setMlir(getRopeMlir()).setNumOutputs(1);

  auto outs = graph->customOp({xT, cosT, sinT}, ropeAttr);
  auto xStride =
      generateStrideFromDim(xDim, getContiguousStrideOrder(xDim.size()));
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

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf}, {cosT, cosBuf}, {sinT, sinBuf}, {outs[0], outBuf}};

  FUSILLI_REQUIRE_ASSIGN(auto workspace,
                         allocateWorkspace(handle, graph->getWorkspaceSize()));

  FUSILLI_REQUIRE_OK(graph->execute(handle, variantPack, workspace));

  std::vector<T> result;
  FUSILLI_REQUIRE_OK(outBuf->read(handle, result));
  return result;
}

} // namespace

TEST_CASE("RoPE: identity rotation (cos=1, sin=0)", "[custom_op][rope]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // B=1, S=1, H=1, D=4 → cos/sin shape [1,1,1,2]
  auto result = executeRope<float>(handle, DataType::Float, /*batch=*/1,
                                   /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
                                   /*xData=*/{1, 2, 3, 4},
                                   /*cosData=*/{1, 1},
                                   /*sinData=*/{0, 0});

  REQUIRE(result.size() == 4);
  REQUIRE(result[0] == 1.0f);
  REQUIRE(result[1] == 2.0f);
  REQUIRE(result[2] == 3.0f);
  REQUIRE(result[3] == 4.0f);
}

TEST_CASE("RoPE: 90-degree rotation (cos=0, sin=1)", "[custom_op][rope]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // x = {1,2,3,4} → x1={1,2}, x2={3,4}
  // out1 = x1*0 - x2*1 = {-3,-4}
  // out2 = x2*0 + x1*1 = {1,2}
  // result = cat({-3,-4},{1,2}) = {-3,-4,1,2}
  auto result = executeRope<float>(handle, DataType::Float, /*batch=*/1,
                                   /*seqLen=*/1, /*heads=*/1, /*headDim=*/4,
                                   /*xData=*/{1, 2, 3, 4},
                                   /*cosData=*/{0, 0},
                                   /*sinData=*/{1, 1});

  REQUIRE(result.size() == 4);
  REQUIRE(result[0] == -3.0f);
  REQUIRE(result[1] == -4.0f);
  REQUIRE(result[2] == 1.0f);
  REQUIRE(result[3] == 2.0f);
}

TEST_CASE("RoPE: larger head_dim (D=8)", "[custom_op][rope]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // B=1, S=1, H=1, D=8 → cos/sin shape [1,1,1,4]
  // 90-degree rotation: x1={1..4}, x2={5..8}
  // out1 = x1*0 - x2*1 = {-5,-6,-7,-8}
  // out2 = x2*0 + x1*1 = {1,2,3,4}
  // result = {-5,-6,-7,-8, 1,2,3,4}
  auto result = executeRope<float>(handle, DataType::Float, /*batch=*/1,
                                   /*seqLen=*/1, /*heads=*/1, /*headDim=*/8,
                                   /*xData=*/{1, 2, 3, 4, 5, 6, 7, 8},
                                   /*cosData=*/{0, 0, 0, 0},
                                   /*sinData=*/{1, 1, 1, 1});

  REQUIRE(result.size() == 8);
  REQUIRE(result[0] == -5.0f);
  REQUIRE(result[1] == -6.0f);
  REQUIRE(result[2] == -7.0f);
  REQUIRE(result[3] == -8.0f);
  REQUIRE(result[4] == 1.0f);
  REQUIRE(result[5] == 2.0f);
  REQUIRE(result[6] == 3.0f);
  REQUIRE(result[7] == 4.0f);
}

TEST_CASE("RoPE: Float16 identity rotation", "[custom_op][rope]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  auto result = executeRope<half>(
      handle, DataType::Half, /*batch=*/1, /*seqLen=*/1, /*heads=*/1,
      /*headDim=*/4,
      /*xData=*/{half(1), half(2), half(3), half(4)},
      /*cosData=*/{half(1), half(1)},
      /*sinData=*/{half(0), half(0)});

  REQUIRE(result.size() == 4);
  REQUIRE(result[0] == half(1));
  REQUIRE(result[1] == half(2));
  REQUIRE(result[2] == half(3));
  REQUIRE(result[3] == half(4));
}
