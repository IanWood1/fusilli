// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef FUSILLI_SAMPLES_CUSTOM_OP_ROPE_UTILS_H
#define FUSILLI_SAMPLES_CUSTOM_OP_ROPE_UTILS_H

#include <string>

// MLIR template for Rotary Position Embedding (RoPE).
//
// Implements standard non-interleaved RoPE:
//   half = D / 2
//   x1, x2 = x[..., :half], x[..., half:]
//   out1 = x1 * cos - x2 * sin
//   out2 = x2 * cos + x1 * sin
//   result = cat(out1, out2, dim=-1)
//
// Tensor convention:
//   x:   [B, S, H, D]     (batch, seq_len, num_heads, head_dim)
//   cos: [1, S, 1, D/2]   (pre-expanded for broadcasting, pre-indexed)
//   sin: [1, S, 1, D/2]   (same)
//   out: [B, S, H, D]     (same shape as x)
//
// Placeholders resolved at emission time by CustomOpNode:
//   {FUNC_NAME}                    -- CustomOpAttr name
//   {IN<i>_TYPE}/{OUT<i>_TYPE}     -- full static tensor type
//   {IN<i>_DTYPE}/{OUT<i>_DTYPE}   -- element type (e.g., "f32")
//   {IN<i>_DIM<j>}/{OUT<i>_DIM<j>} -- single dimension value
//
// %half is the constant {IN1_DIM3} since cos has shape [1, S, 1, D/2].
//
// Subtraction in `out1 = x1*cos - x2*sin` is implemented via
// `torch.aten.add.Tensor(a, b, alpha)` with alpha=-1, since
// `add.Tensor` computes `a + alpha*b`. The `out2` path uses alpha=+1
// for plain addition.
inline std::string getRopeMlir() {
  return R"(
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

    // Split head_dim in half: x1 = x[..., :half], x2 = x[..., half:]
    %x1 = torch.aten.slice.Tensor %x, %int3, %int0, %half, %int1
        : {IN0_TYPE}, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>
    %x2 = torch.aten.slice.Tensor %x, %int3, %half, %int9223372036854775807, %int1
        : {IN0_TYPE}, !torch.int, !torch.int, !torch.int, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>

    // Rotation: out1 = x1*cos - x2*sin (alpha=-1 turns add into subtract)
    %x1_cos = torch.aten.mul.Tensor %x1, %cos
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN1_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %x2_sin = torch.aten.mul.Tensor %x2, %sin
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN2_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %out1 = torch.aten.add.Tensor %x1_cos, %x2_sin, %int_neg1
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>

    // Rotation: out2 = x2*cos + x1*sin (alpha=+1)
    %x2_cos = torch.aten.mul.Tensor %x2, %cos
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN1_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %x1_sin = torch.aten.mul.Tensor %x1, %sin
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{IN0_DTYPE}>, {IN2_TYPE}
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>
    %out2 = torch.aten.add.Tensor %x2_cos, %x1_sin, %int1
        : !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.int
        -> !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>

    // Concatenate halves back: result = cat(out1, out2, dim=3)
    %list = torch.prim.ListConstruct %out1, %out2
        : (!torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>, !torch.vtensor<[{IN0_DIM0},{IN0_DIM1},{IN0_DIM2},{IN1_DIM3}],{OUT0_DTYPE}>)
        -> !torch.list<vtensor>
    %result = torch.aten.cat %list, %int3
        : !torch.list<vtensor>, !torch.int
        -> {OUT0_TYPE}
    return %result : {OUT0_TYPE}
  }
)";
}

#endif // FUSILLI_SAMPLES_CUSTOM_OP_ROPE_UTILS_H
