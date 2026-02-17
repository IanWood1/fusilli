// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// MLIR module with @main entry point, required by the IREE runtime which
// calls `module.main`. This differs from getCustomGraphMLIR() which uses
// @simple_add (suitable for compile-only tests but not runtime execution).
static std::string getCustomGraphMLIR() {
  return R"mlir(
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
)mlir";
}

TEST_CASE("CustomGraph constructor stores name, MLIR, and args",
          "[custom_graph]") {
  std::string mlir = getCustomGraphMLIR();
  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomGraph cg("test_add", mlir, {a, b});

  REQUIRE(cg.getGraphName() == "test_add");
  FUSILLI_REQUIRE_ASSIGN(std::string mlirAsm, cg.getAsm());
  REQUIRE(mlirAsm == mlir);
  REQUIRE(cg.getArgs().size() == 2);
  REQUIRE(cg.getArgs()[0] == a);
  REQUIRE(cg.getArgs()[1] == b);
}

TEST_CASE("CustomGraph compile + execute round-trip", "[custom_graph]") {
  std::string mlir = getCustomGraphMLIR();

  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomGraph cg("simple_add", mlir, {a, b});

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(cg.compile(handle, /*remove=*/true));

  // Allocate input buffers.
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, a, std::vector<float>{1, 2, 3, 4}));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, b, std::vector<float>{5, 6, 7, 8}));

  // Execute — collect returned outputs.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{a, aBuf}, {b, bBuf}};

  std::vector<Buffer> outputs;
  FUSILLI_REQUIRE_OK(cg.execute(handle, variantPack, nullptr, &outputs));

  // The MLIR function returns a new tensor<4xf32>. Read it back from the
  // output buffer returned by the IREE runtime call.
  REQUIRE(outputs.size() == 1);
  std::vector<float> result;
  FUSILLI_REQUIRE_OK(outputs[0].read(handle, result));
  REQUIRE(result.size() == 4);
  REQUIRE(result[0] == 6.0f);
  REQUIRE(result[1] == 8.0f);
  REQUIRE(result[2] == 10.0f);
  REQUIRE(result[3] == 12.0f);
}

TEST_CASE("CustomGraph execute before compile returns NotCompiled error",
          "[custom_graph]") {
  std::string mlir = getCustomGraphMLIR();

  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomGraph cg("not_compiled", mlir, {a, b});

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  // Allocate buffers.
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, a, std::vector<float>{1, 2, 3, 4}));
  FUSILLI_REQUIRE_ASSIGN(
      auto bBuf,
      allocateBufferOfType(handle, b, std::vector<float>{5, 6, 7, 8}));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{a, aBuf}, {b, bBuf}};

  auto status = cg.execute(handle, variantPack);
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotCompiled);
}

TEST_CASE("CustomGraph execute with missing variantPack entry returns error",
          "[custom_graph]") {
  std::string mlir = getCustomGraphMLIR();

  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomGraph cg("missing_arg", mlir, {a, b});

  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));
  FUSILLI_REQUIRE_OK(cg.compile(handle, /*remove=*/true));

  // Only provide one of the two required args.
  FUSILLI_REQUIRE_ASSIGN(
      auto aBuf,
      allocateBufferOfType(handle, a, std::vector<float>{1, 2, 3, 4}));

  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {{a, aBuf}};

  auto status = cg.execute(handle, variantPack);
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::VariantPackError);
}

TEST_CASE("CustomGraph getCompiledArtifact cache behavior", "[custom_graph]") {
  FUSILLI_REQUIRE_ASSIGN(Handle handle, Handle::create(kDefaultBackend));

  std::string mlir = getCustomGraphMLIR();
  auto a = std::make_shared<TensorAttr>(
      TensorAttr().setName("a").setDim({4}).setStride({1}).setDataType(
          DataType::Float));
  auto b = std::make_shared<TensorAttr>(
      TensorAttr().setName("b").setDim({4}).setStride({1}).setDataType(
          DataType::Float));

  CustomGraph cg("cache_test", mlir, {a, b});

  // First compilation should generate cache.
  std::optional<bool> reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(
      cg.getCompiledArtifact(handle, mlir, /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(reCompiled.value());

  // Second call should hit cache.
  reCompiled = std::nullopt;
  FUSILLI_REQUIRE_OK(
      cg.getCompiledArtifact(handle, mlir, /*remove=*/true, &reCompiled));
  REQUIRE(reCompiled.has_value());
  REQUIRE(!reCompiled.value());
}
