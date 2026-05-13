// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace fusilli;

TEST_CASE("LayerNormNode getName correctly propagates the attribute name",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setName("foo_layernorm");

  LayerNormNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_layernorm");
}

TEST_CASE("LayerNormNode getType returns correct type", "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  LayerNormNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::LayerNorm);
}

TEST_CASE("LayerNormBwdNode getName correctly propagates the attribute name",
          "[layernorm_bwd_node]") {
  Context ctx;
  LayernormBwdAttr attr;
  attr.setName("foo_layernorm_bwd");

  LayerNormBwdNode node(std::move(attr), ctx);
  REQUIRE(node.getName() == "foo_layernorm_bwd");
}

TEST_CASE("LayerNormBwdNode getType returns correct type",
          "[layernorm_bwd_node]") {
  Context ctx;
  LayernormBwdAttr attr;

  LayerNormBwdNode node(std::move(attr), ctx);
  REQUIRE(node.getType() == INode::Type::LayerNormBwd);
}

TEST_CASE("LayerNormNode preValidateNode detects missing attributes",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  SECTION("Forward phase not set") {
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm forward phase not set");
  }

  SECTION("Input X missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm input tensor X not set");
  }

  SECTION("Output Y missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm output tensor Y not set");
  }

  SECTION("Epsilon missing") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE);
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm epsilon not set");
  }

  SECTION("All required attributes present for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for INFERENCE forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("Extra output MEAN for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor MEAN should not be set");
  }

  SECTION("Extra output INV_VARIANCE for INFERENCE forward phase") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor INV_VARIANCE should not be set");
  }

  SECTION("Output MEAN missing for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() == "LayerNorm output tensor MEAN not set");
  }

  SECTION("Output INV_VARIANCE missing for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() ==
            "LayerNorm output tensor INV_VARIANCE not set");
  }

  SECTION("All required attributes present for TRAINING forward phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }

  SECTION("All required and optional attributes present for TRAINING forward "
          "phase") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 3}).setStride({3, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 3}).setStride({3, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({2, 1}).setStride({1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
  }
}

TEST_CASE("LayerNormBwdNode validation and inference", "[layernorm_bwd_node]") {
  Context ctx;
  constexpr int64_t n = 2, c = 3, h = 4;

  auto makeAttr = [=] {
    LayernormBwdAttr attr;
    attr.setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setDY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h}).setStride({c * h, h, 1})));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h}).setStride({c * h, h, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, h}).setStride({c * h, h, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setDX(std::make_shared<TensorAttr>());
    attr.setDSCALE(std::make_shared<TensorAttr>());
    attr.setDBIAS(std::make_shared<TensorAttr>());
    return attr;
  };

  SECTION("All required attributes present") {
    LayerNormBwdNode node(makeAttr(), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    FUSILLI_REQUIRE_OK(node.postValidateNode());

    REQUIRE(node.layernormBwdAttr.getDX()->getDim() ==
            std::vector<int64_t>{n, c, h});
    REQUIRE(node.layernormBwdAttr.getDX()->getStride() ==
            std::vector<int64_t>{c * h, h, 1});
    REQUIRE(node.layernormBwdAttr.getDSCALE()->getDim() ==
            std::vector<int64_t>{1, c, h});
    REQUIRE(node.layernormBwdAttr.getDSCALE()->getStride() ==
            std::vector<int64_t>{c * h, h, 1});
    REQUIRE(node.layernormBwdAttr.getDBIAS()->getDim() ==
            std::vector<int64_t>{1, c, h});
    REQUIRE(node.layernormBwdAttr.getDBIAS()->getStride() ==
            std::vector<int64_t>{c * h, h, 1});
  }

  SECTION("Missing DY is rejected") {
    auto attr = makeAttr();
    attr.setDY(nullptr);
    LayerNormBwdNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(status.getMessage() ==
            "LayerNorm backward input tensor DY not set");
  }

  SECTION("Incorrect DSCALE shape is rejected") {
    auto attr = makeAttr();
    attr.setDSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, h}).setStride({c * h, h, 1})));
    LayerNormBwdNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm backward tensor DSCALE must have shape as tensor X "
            "with single batch");
  }
}

TEST_CASE(
    "LayerNormNode inferPropertiesNode when output tensors are fully specified",
    "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setMEAN(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1}).setStride({1, 1})));
  attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, 1}).setStride({1, 1})));

  LayerNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.layernormAttr.getY();
  auto mT = node.layernormAttr.getMEAN();
  auto vT = node.layernormAttr.getINV_VARIANCE();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(mT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(mT->getStride() == std::vector<int64_t>{1, 1});
  REQUIRE(vT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(vT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE(
    "LayerNormNode inferPropertiesNode when output tensors are under-specified",
    "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setForwardPhase(NormFwdPhase::TRAINING);

  int64_t n = 2, c = 5;

  attr.setX(std::make_shared<TensorAttr>(
      TensorAttr().setDim({n, c}).setStride({c, 1})));
  attr.setY(std::make_shared<TensorAttr>());
  attr.setMEAN(std::make_shared<TensorAttr>());
  attr.setINV_VARIANCE(std::make_shared<TensorAttr>());

  LayerNormNode node(std::move(attr), ctx);
  FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

  auto yT = node.layernormAttr.getY();
  auto mT = node.layernormAttr.getMEAN();
  auto vT = node.layernormAttr.getINV_VARIANCE();
  REQUIRE(yT->getDim() == std::vector<int64_t>{n, c});
  REQUIRE(yT->getStride() == std::vector<int64_t>{c, 1});
  REQUIRE(mT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(mT->getStride() == std::vector<int64_t>{1, 1});
  REQUIRE(vT->getDim() == std::vector<int64_t>{n, 1});
  REQUIRE(vT->getStride() == std::vector<int64_t>{1, 1});
}

TEST_CASE(
    "LayerNormNode inferPropertiesNode when SCALE/BIAS tensors are unspecified",
    "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;
  attr.setForwardPhase(NormFwdPhase::INFERENCE)
      .setEpsilon(std::make_shared<TensorAttr>(1e-5f));

  int64_t n = 2, c = 5, d = 10;

  SECTION("SCALE shape and strides are unspecified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(TensorAttr()));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.layernormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, d, 1});
  }

  SECTION("SCALE shape and strides are partically specified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setSCALE(std::make_shared<TensorAttr>(TensorAttr().setDim({1, c, d})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.layernormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }

  SECTION("SCALE shape and strides are set") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, d}).setStride({c * d, 1, c})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto sT = node.layernormAttr.getSCALE();
    REQUIRE(sT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(sT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }

  SECTION("BIAS shape and strides are unspecified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(TensorAttr()));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto bT = node.layernormAttr.getBIAS();
    REQUIRE(bT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(bT->getStride() == std::vector<int64_t>{c * d, d, 1});
  }

  SECTION("BIAS shape and strides are partically specified") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setBIAS(std::make_shared<TensorAttr>(TensorAttr().setDim({1, c, d})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto bT = node.layernormAttr.getBIAS();
    REQUIRE(bT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(bT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }

  SECTION("BIAS shape and strides are set") {
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, 1, c})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, c, d}).setStride({c * d, 1, c})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());

    auto bT = node.layernormAttr.getBIAS();
    REQUIRE(bT->getDim() == std::vector<int64_t>{1, c, d});
    REQUIRE(bT->getStride() == std::vector<int64_t>{c * d, 1, c});
  }
}

TEST_CASE("LayerNormNode shape checks on SCALE and BIAS tensors",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  int64_t n = 2, c = 3, d = 4;

  SECTION("Incorrect SCALE shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setSCALE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm input tensor SCALE must have shape as "
            "tensor X with single batch");
  }

  SECTION("Incorrect BIAS shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setBIAS(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>());
    LayerNormNode node(std::move(attr), ctx);

    auto status = node.preValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm input tensor BIAS must have shape as "
            "tensor X with single batch");
  }
}

TEST_CASE("LayerNormNode postValidateNode detects incorrect shapes and strides",
          "[layernorm_node]") {
  Context ctx;
  LayernormAttr attr;

  int64_t n = 2, c = 3, d = 4;

  SECTION("Output Y has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n + 1, c, d}).setStride({c * d, d, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(
        status.getMessage() ==
        "LayerNorm output Y tensor must have the same shape as input X tensor");
  }

  SECTION("Output Y has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::INFERENCE)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(TensorAttr()
                                               .setDim({n, c, d})
                                               .setStride({d, c * d, 1})
                                               .setName("Y_invalid_layout")));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::NotImplemented);
    REQUIRE(status.getMessage() ==
            "Tensor 'Y_invalid_layout' is neither contiguous nor channels-last "
            "as defined by its stride");
  }

  SECTION("Output MEAN has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "Layernorm output MEAN tensor must have shape [B, 1, ..., 1] with "
            "rank equal to input X tensor's rank, and batch dimension equal "
            "to input X tensor's batch dimension");
  }

  SECTION("Output MEAN has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({n, 1, 1}).setName(
            "MEAN_invalid_layout")));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output MEAN tensor must have unit strides");
  }

  SECTION("Output INV_VARIANCE has incorrect shape") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({1, 1, 1}).setStride({1, 1, 1})));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output INV_VARIANCE tensor must have "
            "shape [B, 1, ..., 1] with  rank equal to "
            "input X tensor's rank, and batch dimension equal "
            "to input X tensor's batch dimension");
  }

  SECTION("Output INV_VARIANCE has incorrect stride") {
    attr.setForwardPhase(NormFwdPhase::TRAINING)
        .setEpsilon(std::make_shared<TensorAttr>(1e-5f));
    attr.setX(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setY(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, c, d}).setStride({c * d, d, 1})));
    attr.setMEAN(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, 1})));
    attr.setINV_VARIANCE(std::make_shared<TensorAttr>(
        TensorAttr().setDim({n, 1, 1}).setStride({1, 1, n}).setName(
            "INV_VARIANCE_invalid_layout")));
    LayerNormNode node(std::move(attr), ctx);

    FUSILLI_REQUIRE_OK(node.preValidateNode());
    FUSILLI_REQUIRE_OK(node.inferPropertiesNode());
    auto status = node.postValidateNode();
    REQUIRE(isError(status));
    REQUIRE(status.getCode() == ErrorCode::InvalidAttribute);
    REQUIRE(status.getMessage() ==
            "LayerNorm output INV_VARIANCE tensor must have unit strides");
  }
}
