// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Utilities for MLIR assembly emission. This file provides core types and
// generic helpers that are independent of any specific node type.
//
// The central abstraction is `AsmValue`, which pairs an SSA name with its
// MLIR type — mirroring the concept of an MLIR Value. Helpers return
// `AsmEmission` structs that bundle prerequisite MLIR ops with the
// resulting `AsmValue`, so callers never need to independently reconstruct
// the names that those ops produce.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_ASM_UTILS_H
#define FUSILLI_SUPPORT_ASM_UTILS_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/support/extras.h"

#include <cstddef>
#include <cstdint>
#include <format> // C++20
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
// Core types
//===----------------------------------------------------------------------===//

/// An MLIR SSA value: a name and its type.
/// Models the result of any op that produces a value the compute op
/// references.
struct AsmValue {
  std::string name; // e.g. "%X_conv_fprop_perm"
  std::string type; // e.g. "!torch.vtensor<[4,16,8,8],f32>"
};

/// Prerequisite MLIR ops plus the resulting value they produce.
/// Every emitter helper (permute, scalar extract, list construct, none
/// decl) returns this: the MLIR text to emit, and the value the compute
/// op should reference.
struct AsmEmission {
  std::string ops; // MLIR text for prerequisite ops (may be empty)
  AsmValue value;  // The SSA value produced by those ops
};

//===----------------------------------------------------------------------===//
// Joining utilities
//===----------------------------------------------------------------------===//

/// Join a list of AsmValues into comma-separated name and type strings.
/// Returns {names, types} for use in compute op formatting.
inline std::pair<std::string, std::string>
joinOperands(const std::vector<AsmValue> &values) {
  std::ostringstream names, types;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      names << ", ";
      types << ", ";
    }
    names << values[i].name;
    types << values[i].type;
  }
  return {names.str(), types.str()};
}

//===----------------------------------------------------------------------===//
// Permute emission
//===----------------------------------------------------------------------===//

// Emits permute ops for a tensor in MLIR assembly format.
//
// When isInput=true (physical-to-logical permute):
//   - Operand: {name} with physical layout
//   - Result:  {name}_{suffix}_perm with logical layout
//
// When isInput=false (logical-to-physical permute):
//   - Operand: {name}_{suffix}_perm with logical layout
//   - Result:  {name} with physical layout
//
// The returned AsmEmission::value is the compute-side SSA value — i.e.
// the name/type that the main op should use as its operand (for input
// permutes) or result (for output permutes).
//
// The suffix is used to ensure unique SSA names when the same tensor is
// used by multiple different operations in a graph.
inline AsmEmission emitPermute(const std::shared_ptr<TensorAttr> &tensor,
                               const std::string &prefix,
                               const std::string &suffix, bool isInput) {
  std::ostringstream oss;

  // Get permute order based on direction.
  std::vector<int64_t> permuteOrder =
      isInput ? tensor->getPhysicalToLogicalPermuteOrder()
              : tensor->getLogicalToPhysicalPermuteOrder();

  // Emit `torch.constant.int` ops for each element and a
  // `torch.prim.ListConstruct` wrapping them.
  {
    std::vector<std::string> ssaValueNames;

    for (size_t i = 0; i < permuteOrder.size(); ++i) {
      std::string ssaValueName =
          "%" + prefix + "_val_" + std::to_string(i) + "_" + suffix;
      oss << ssaValueName << " = torch.constant.int " << permuteOrder[i]
          << "\n    ";
      ssaValueNames.push_back(ssaValueName);
    }

    oss << "%" + prefix + "_" + suffix << " = torch.prim.ListConstruct ";
    interleave(
        ssaValueNames.begin(), ssaValueNames.end(),
        [&](const std::string &name) { oss << name; }, [&] { oss << ", "; });
    oss << " : (";
    interleave(
        ssaValueNames.begin(), ssaValueNames.end(),
        [&](const std::string &name) { oss << "!torch.int"; },
        [&] { oss << ", "; });
    oss << ") -> !torch.list<int>\n";
  }

  // Include the suffix in permuted tensor names to ensure uniqueness when
  // the same tensor is used by multiple operations. For input permutes, the
  // operand is the original tensor (no suffix needed), but the result gets
  // the suffix. For output permutes, the operand has the suffix (from the
  // main op result), but the result is the final tensor (no suffix needed).
  std::string resultName =
      tensor->getValueNameAsm() + (isInput ? "_" + suffix + "_perm" : "");
  std::string operandName =
      tensor->getValueNameAsm() + (isInput ? "" : "_" + suffix + "_perm");
  std::string fromType = tensor->getTensorTypeAsm(
      /*isValueTensor=*/true, /*useLogicalDims=*/!isInput);
  std::string toType = tensor->getTensorTypeAsm(
      /*isValueTensor=*/true, /*useLogicalDims=*/isInput);

  constexpr std::string_view schema = R"(
    {0} = torch.aten.permute {1}, {2} : {3}, !torch.list<int> -> {4}
  )";
  oss << std::format(schema,
                     resultName,                  // {0}
                     operandName,                 // {1}
                     "%" + prefix + "_" + suffix, // {2}
                     fromType,                    // {3}
                     toType                       // {4}
  );

  // The compute-side value is always the one with "_suffix_perm":
  //   input:  resultName  = %X_suffix_perm  (compute op reads this)
  //   output: operandName = %Y_suffix_perm  (compute op writes this)
  std::string computeName = isInput ? resultName : operandName;
  std::string computeType = tensor->getTensorTypeAsm(
      /*isValueTensor=*/true, /*useLogicalDims=*/true);

  return {oss.str(), {std::move(computeName), std::move(computeType)}};
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_ASM_UTILS_H
