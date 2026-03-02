// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s

// Verifies RoPE custom op ASM emission:
//   - Module-scope func.func private @rope with 3 inputs, 1 output
//   - slice.Tensor, mul.Tensor, add.Tensor, prim.ListConstruct, cat ops
//   - func.call @rope in main with static-shape operands

// clang-format off
//
// CHECK:       module @module {
// CHECK:         func.func private @rope(
// CHECK-NEXT:      %x: !torch.vtensor<[1,1,1,4],f32>,
// CHECK-NEXT:      %cos: !torch.vtensor<[1,1,1,2],f32>,
// CHECK-NEXT:      %sin: !torch.vtensor<[1,1,1,2],f32>)
// CHECK-NEXT:      -> !torch.vtensor<[1,1,1,4],f32> {
// CHECK:           %int0 = torch.constant.int 0
// CHECK:           %int1 = torch.constant.int 1
// CHECK:           %int3 = torch.constant.int 3
// CHECK:           %int_neg1 = torch.constant.int -1
// CHECK:           %int9223372036854775807 = torch.constant.int 9223372036854775807
// CHECK:           %half = torch.constant.int 2
// CHECK:           %x1 = torch.aten.slice.Tensor %x, %int3, %int0, %half, %int1
// CHECK:           %x2 = torch.aten.slice.Tensor %x, %int3, %half, %int9223372036854775807, %int1
// CHECK:           %x1_cos = torch.aten.mul.Tensor %x1, %cos
// CHECK:           %x2_sin = torch.aten.mul.Tensor %x2, %sin
// CHECK:           %out1 = torch.aten.add.Tensor %x1_cos, %x2_sin, %int_neg1
// CHECK:           %x2_cos = torch.aten.mul.Tensor %x2, %cos
// CHECK:           %x1_sin = torch.aten.mul.Tensor %x1, %sin
// CHECK:           %out2 = torch.aten.add.Tensor %x2_cos, %x1_sin, %int1
// CHECK:           %list = torch.prim.ListConstruct %out1, %out2
// CHECK:           %result = torch.aten.cat %list, %int3
// CHECK:           return %result
// CHECK:         }
// CHECK:         func.func @main(
// CHECK-SAME:      %rope_OUT_0_: !torch.tensor<[1,1,1,4],f32>
// CHECK-SAME:      %cos: !torch.vtensor<[1,1,1,2],f32>
// CHECK-SAME:      %sin: !torch.vtensor<[1,1,1,2],f32>
// CHECK-SAME:      %x: !torch.vtensor<[1,1,1,4],f32>
// CHECK:           %x_rope_i0_perm = torch.aten.permute %x
// CHECK:           %cos_rope_i1_perm = torch.aten.permute %cos
// CHECK:           %sin_rope_i2_perm = torch.aten.permute %sin
// CHECK:           %rope_OUT_0_rope_perm = func.call @rope(%x_rope_i0_perm, %cos_rope_i1_perm, %sin_rope_i2_perm)
// CHECK-SAME:        : (!torch.vtensor<[1,1,1,4],f32>, !torch.vtensor<[1,1,1,2],f32>, !torch.vtensor<[1,1,1,2],f32>) -> !torch.vtensor<[1,1,1,4],f32>
// CHECK:           %rope_OUT_0 = torch.aten.permute %rope_OUT_0_rope_perm
// CHECK:           torch.overwrite.tensor.contents %rope_OUT_0 overwrites %rope_OUT_0_
// CHECK:           return
// CHECK:         }
// CHECK:       }
//
// clang-format on

#include <fusilli.h>

#include "utils.h"

#include <iostream>
#include <string>

using namespace fusilli;

int main() {
  Graph g;
  g.setName("rope_asm_emitter").setIODataType(DataType::Float);

  auto x = g.tensor(TensorAttr()
                        .setName("x")
                        .setDim({1, 1, 1, 4})
                        .setStride({4, 4, 4, 1})
                        .setDataType(DataType::Float));
  auto cos = g.tensor(TensorAttr()
                          .setName("cos")
                          .setDim({1, 1, 1, 2})
                          .setStride({2, 2, 2, 1})
                          .setDataType(DataType::Float));
  auto sin = g.tensor(TensorAttr()
                          .setName("sin")
                          .setDim({1, 1, 1, 2})
                          .setStride({2, 2, 2, 1})
                          .setDataType(DataType::Float));

  std::string ropeMlir = R"(
  func.func private @{FUNC_NAME}(
      %x: {IN0_TYPE},
      %cos: {IN1_TYPE},
      %sin: {IN2_TYPE})
      -> {OUT0_TYPE} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int_neg1 = torch.constant.int -1
    %int9223372036854775807 = torch.constant.int 9223372036854775807

    %half = torch.constant.int {IN1_DIM3}

    %x1 = torch.aten.slice.Tensor %x, %int3, %int0, %half, %int1
        : {IN0_TYPE}, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>
    %x2 = torch.aten.slice.Tensor %x, %int3, %half, %int9223372036854775807, %int1
        : {IN0_TYPE}, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>

    // out1 = x1*cos - x2*sin (alpha=-1 on add.Tensor turns add into subtract).
    %x1_cos = torch.aten.mul.Tensor %x1, %cos
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN1_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %x2_sin = torch.aten.mul.Tensor %x2, %sin
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN2_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %out1 = torch.aten.add.Tensor %x1_cos, %x2_sin, %int_neg1
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>

    %x2_cos = torch.aten.mul.Tensor %x2, %cos
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN1_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %x1_sin = torch.aten.mul.Tensor %x1, %sin
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN2_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %out2 = torch.aten.add.Tensor %x2_cos, %x1_sin, %int1
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>

    %list = torch.prim.ListConstruct %out1, %out2
        : (!torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>)
        -> !torch.list<vtensor>
    %result = torch.aten.cat %list, %int3
        : !torch.list<vtensor>, !torch.int
        -> {OUT0_TYPE}
    return %result : {OUT0_TYPE}
  }
)";

  CustomOpAttr ropeAttr;
  ropeAttr.setName("rope").setMlir(ropeMlir).setNumOutputs(1);

  auto outs = g.customOp({x, cos, sin}, ropeAttr);
  outs[0]
      ->setDim({1, 1, 1, 4})
      .setStride({4, 4, 4, 1})
      .setDataType(DataType::Float)
      .setOutput(true);

  auto status = g.validate();
  if (isError(status)) {
    std::cerr << "Validation failed: " << status << std::endl;
    return 1;
  }

  auto asmOrErr = g.emitAsm();
  if (isError(asmOrErr)) {
    std::cerr << "ASM emission failed: " << asmOrErr << std::endl;
    return 1;
  }

  auto indentErr = checkMlirIndentation(*asmOrErr);
  if (isError(indentErr)) {
    std::cerr << "Indentation check failed: " << indentErr << std::endl;
    return 1;
  }

  std::cout << *asmOrErr << std::endl;
  return 0;
}
