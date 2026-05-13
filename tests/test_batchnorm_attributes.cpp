// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>
#include <memory>

using namespace fusilli;

TEST_CASE("BatchnormAttr default constructor", "[batchnorm_attr]") {
  BatchnormAttr attr;
  REQUIRE(attr.getForwardPhase() == NormFwdPhase::NOT_SET);
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("BatchnormAttr setters and getters", "[batchnorm_attr]") {
  BatchnormAttr attr;

  attr.setForwardPhase(NormFwdPhase::INFERENCE);
  REQUIRE(attr.getForwardPhase() == NormFwdPhase::INFERENCE);

  auto x = std::make_shared<TensorAttr>(1.0f);
  auto s = std::make_shared<TensorAttr>(2.0f);
  auto b = std::make_shared<TensorAttr>(3.0f);
  auto mean = std::make_shared<TensorAttr>(4.0f);
  auto var = std::make_shared<TensorAttr>(5.0f);
  auto e = std::make_shared<TensorAttr>(1e-5f);
  auto m = std::make_shared<TensorAttr>(0.1f);
  auto y = std::make_shared<TensorAttr>(6.0f);
  auto sm = std::make_shared<TensorAttr>(7.0f);
  auto siv = std::make_shared<TensorAttr>(8.0f);

  attr.setX(x).setSCALE(s).setBIAS(b).setMEAN(mean).setVAR(var);
  attr.setEpsilon(e).setMomentum(m);
  attr.setY(y).setSAVED_MEAN(sm).setSAVED_INV_VARIANCE(siv);

  REQUIRE(attr.inputs.size() == 7);
  REQUIRE(attr.outputs.size() == 3);

  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getSCALE() == s);
  REQUIRE(attr.getBIAS() == b);
  REQUIRE(attr.getMEAN() == mean);
  REQUIRE(attr.getVAR() == var);
  REQUIRE(attr.getEpsilon() == e);
  REQUIRE(attr.getMomentum() == m);
  REQUIRE(attr.getY() == y);
  REQUIRE(attr.getSAVED_MEAN() == sm);
  REQUIRE(attr.getSAVED_INV_VARIANCE() == siv);
}

TEST_CASE("BatchnormAttr training phase setters", "[batchnorm_attr]") {
  BatchnormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);
  REQUIRE(attr.getForwardPhase() == NormFwdPhase::TRAINING);
}

TEST_CASE("BatchnormAttr optional tensors default to null",
          "[batchnorm_attr]") {
  BatchnormAttr attr;
  REQUIRE(attr.getX() == nullptr);
  REQUIRE(attr.getSCALE() == nullptr);
  REQUIRE(attr.getBIAS() == nullptr);
  REQUIRE(attr.getMEAN() == nullptr);
  REQUIRE(attr.getVAR() == nullptr);
  REQUIRE(attr.getEpsilon() == nullptr);
  REQUIRE(attr.getMomentum() == nullptr);
  REQUIRE(attr.getY() == nullptr);
  REQUIRE(attr.getSAVED_MEAN() == nullptr);
  REQUIRE(attr.getSAVED_INV_VARIANCE() == nullptr);
}

TEST_CASE("BatchnormBwdAttr default constructor", "[batchnorm_bwd_attr]") {
  BatchnormBwdAttr attr;
  REQUIRE(attr.inputs.empty());
  REQUIRE(attr.outputs.empty());
}

TEST_CASE("BatchnormBwdAttr setters and getters", "[batchnorm_bwd_attr]") {
  BatchnormBwdAttr attr;

  auto dy = std::make_shared<TensorAttr>(1.0f);
  auto x = std::make_shared<TensorAttr>(2.0f);
  auto scale = std::make_shared<TensorAttr>(3.0f);
  auto mean = std::make_shared<TensorAttr>(4.0f);
  auto invVariance = std::make_shared<TensorAttr>(5.0f);
  auto dx = std::make_shared<TensorAttr>(6.0f);
  auto dscale = std::make_shared<TensorAttr>(7.0f);
  auto dbias = std::make_shared<TensorAttr>(8.0f);

  attr.setDY(dy).setX(x).setSCALE(scale).setMEAN(mean).setINV_VARIANCE(
      invVariance);
  attr.setDX(dx).setDSCALE(dscale).setDBIAS(dbias);

  REQUIRE(attr.inputs.size() == 5);
  REQUIRE(attr.outputs.size() == 3);

  REQUIRE(attr.getDY() == dy);
  REQUIRE(attr.getX() == x);
  REQUIRE(attr.getSCALE() == scale);
  REQUIRE(attr.getMEAN() == mean);
  REQUIRE(attr.getINV_VARIANCE() == invVariance);
  REQUIRE(attr.getDX() == dx);
  REQUIRE(attr.getDSCALE() == dscale);
  REQUIRE(attr.getDBIAS() == dbias);
}

TEST_CASE("BatchnormBwdAttr optional tensors default to null",
          "[batchnorm_bwd_attr]") {
  BatchnormBwdAttr attr;
  REQUIRE(attr.getDY() == nullptr);
  REQUIRE(attr.getX() == nullptr);
  REQUIRE(attr.getSCALE() == nullptr);
  REQUIRE(attr.getMEAN() == nullptr);
  REQUIRE(attr.getINV_VARIANCE() == nullptr);
  REQUIRE(attr.getDX() == nullptr);
  REQUIRE(attr.getDSCALE() == nullptr);
  REQUIRE(attr.getDBIAS() == nullptr);
}
