// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `CustomGraph` class which takes
// pre-written MLIR directly and compiles/executes it through the IREE backend.
// Unlike `Graph`, it does not build MLIR from a node tree.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_CUSTOM_GRAPH_H
#define FUSILLI_GRAPH_CUSTOM_GRAPH_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/graph/graph_base.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fusilli {

class CustomGraph : public GraphCRTP<CustomGraph> {
public:
  // Constructs a CustomGraph from pre-written MLIR.
  //
  // `name`: Identifier used for cache path keys.
  // `mlir`: MLIR source string (e.g., from torch-mlir export).
  // `args`: TensorAttr objects matching the MLIR func.func @main() signature
  //         order. These serve as keys in the variantPack for execute().
  CustomGraph(std::string name, std::string mlir,
              std::vector<std::shared_ptr<TensorAttr>> args)
      : name_(std::move(name)), mlir_(std::move(mlir)), args_(std::move(args)) {
  }

  ErrorObject compile(const Handle &handle, bool remove = false) {
    return compileImpl(handle, remove);
  }

  // `workspace` — transient storage buffer, see GraphCRTP::getWorkspaceSize().
  // `outputs` — if non-null, receives any buffer views returned by the IREE
  // function call. See `GraphCRTP::executeImpl` for details.
  ErrorObject
  execute(const Handle &handle,
          const std::unordered_map<std::shared_ptr<TensorAttr>,
                                   std::shared_ptr<Buffer>> &variantPack,
          const std::shared_ptr<Buffer> &workspace = nullptr,
          std::vector<Buffer> *outputs = nullptr) const {
    return executeImpl(handle, variantPack, workspace, outputs);
  }

  // CRTP interface for GraphCRTP<CustomGraph>.
  const std::string &getGraphName() const { return name_; }
  ErrorOr<std::string> getAsm() const { return ok(mlir_); }

  // Populates IREE runtime call inputs from the variantPack for CustomGraph.
  // Sequential push in arg order matching the MLIR func.func @main() signature.
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject populateCallInputs(
      iree_runtime_call_t &call,
      const std::unordered_map<std::shared_ptr<TensorAttr>,
                               std::shared_ptr<Buffer>> &variantPack) const;

  const std::vector<std::shared_ptr<TensorAttr>> &getArgs() const {
    return args_;
  }

private:
  std::string name_;
  std::string mlir_;
  std::vector<std::shared_ptr<TensorAttr>> args_;
};

} // namespace fusilli

#endif // FUSILLI_GRAPH_CUSTOM_GRAPH_H
