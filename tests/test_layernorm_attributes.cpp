// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <vector>

using namespace fusilli;

TEST_CASE("LayernormAttr default constructor", "[layernorm_attr]") {
  LayernormAttr attr;
  REQUIRE(attr.getForwardPhase() == NormFwdPhase::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("LayernormAttr setters and getters", "[layernorm_attr]") {
  LayernormAttr attr;

  NormFwdPhase phase = NormFwdPhase::INFERENCE;

  attr.setForwardPhase(phase);

  REQUIRE(attr.getForwardPhase() == phase);

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto s = std::make_shared<TensorAttr>(2.0f);
  auto b = std::make_shared<TensorAttr>(3.0f);
  auto e = std::make_shared<TensorAttr>(4.0f);
  auto y = std::make_shared<TensorAttr>(5.0f);
  auto m = std::make_shared<TensorAttr>(6.0f);
  auto v = std::make_shared<TensorAttr>(7.0f);

  attr.setX(x).setSCALE(s).setBIAS(b).setY(y).setMEAN(m).setINV_VARIANCE(v);
  attr.setEpsilon(e);

  REQUIRE(attr.inputs.size() == 4);
  REQUIRE(attr.outputs.size() == 3);

#define CHECK_TENSOR_PROPERTIES(NAME, TENSOR)                                  \
  REQUIRE(attr.get##NAME() == TENSOR);                                         \
  REQUIRE(attr.get##NAME()->getDataType() == DataType::Float);                 \
  REQUIRE(attr.get##NAME()->getDim() == std::vector<int64_t>{1});              \
  REQUIRE(attr.get##NAME()->getStride() == std::vector<int64_t>{1});           \
  REQUIRE(attr.get##NAME()->isScalar() == true);                               \
  REQUIRE(attr.get##NAME()->isVirtual() == false)

  CHECK_TENSOR_PROPERTIES(X, x);
  CHECK_TENSOR_PROPERTIES(SCALE, s);
  CHECK_TENSOR_PROPERTIES(BIAS, b);
  CHECK_TENSOR_PROPERTIES(Y, y);
  CHECK_TENSOR_PROPERTIES(MEAN, m);
  CHECK_TENSOR_PROPERTIES(INV_VARIANCE, v);

  CHECK_TENSOR_PROPERTIES(Epsilon, e);

#undef CHECK_TENSOR_PROPERTIES
}

TEST_CASE("LayernormBwdAttr default constructor", "[layernorm_bwd_attr]") {
  LayernormBwdAttr attr;
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("LayernormBwdAttr setters and getters", "[layernorm_bwd_attr]") {
  LayernormBwdAttr attr;

  auto dy = std::make_shared<TensorAttr>(1.0f);
  auto x = std::make_shared<TensorAttr>(2.0f);
  auto s = std::make_shared<TensorAttr>(3.0f);
  auto m = std::make_shared<TensorAttr>(4.0f);
  auto v = std::make_shared<TensorAttr>(5.0f);
  auto e = std::make_shared<TensorAttr>(6.0f);
  auto dx = std::make_shared<TensorAttr>(7.0f);
  auto ds = std::make_shared<TensorAttr>(8.0f);
  auto db = std::make_shared<TensorAttr>(9.0f);

  attr.setDY(dy)
      .setX(x)
      .setSCALE(s)
      .setMEAN(m)
      .setINV_VARIANCE(v)
      .setDX(dx)
      .setDSCALE(ds)
      .setDBIAS(db);
  attr.setEpsilon(e);

  REQUIRE(attr.inputs.size() == 6);
  REQUIRE(attr.outputs.size() == 3);

#define CHECK_TENSOR_PROPERTIES(NAME, TENSOR)                                  \
  REQUIRE(attr.get##NAME() == TENSOR);                                         \
  REQUIRE(attr.get##NAME()->getDataType() == DataType::Float);                 \
  REQUIRE(attr.get##NAME()->getDim() == std::vector<int64_t>{1});              \
  REQUIRE(attr.get##NAME()->getStride() == std::vector<int64_t>{1});           \
  REQUIRE(attr.get##NAME()->isScalar() == true);                               \
  REQUIRE(attr.get##NAME()->isVirtual() == false)

  CHECK_TENSOR_PROPERTIES(DY, dy);
  CHECK_TENSOR_PROPERTIES(X, x);
  CHECK_TENSOR_PROPERTIES(SCALE, s);
  CHECK_TENSOR_PROPERTIES(MEAN, m);
  CHECK_TENSOR_PROPERTIES(INV_VARIANCE, v);
  CHECK_TENSOR_PROPERTIES(DX, dx);
  CHECK_TENSOR_PROPERTIES(DSCALE, ds);
  CHECK_TENSOR_PROPERTIES(DBIAS, db);

  CHECK_TENSOR_PROPERTIES(Epsilon, e);

#undef CHECK_TENSOR_PROPERTIES
}
