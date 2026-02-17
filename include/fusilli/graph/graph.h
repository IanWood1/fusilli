// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `Graph` class which derives from the
// `INode` class (like other nodes) and `GraphCRTP<Graph>` for compile/execute
// infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_GRAPH_H
#define FUSILLI_GRAPH_GRAPH_H

#include "fusilli/attributes/common.h"
#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/attributes/layernorm_attributes.h"
#include "fusilli/attributes/matmul_attributes.h"
#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/reduction_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/graph/context.h"
#include "fusilli/graph/graph_base.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/layernorm_node.h"
#include "fusilli/node/matmul_node.h"
#include "fusilli/node/node.h"
#include "fusilli/node/pointwise_node.h"
#include "fusilli/node/reduction_node.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fusilli {

class Graph : public INode, public GraphCRTP<Graph> {
public:
  Graph() : INode(Context{}) {}

  // Validates the graph for correctness and infers missing properties.
  ErrorObject validate() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating Graph");
    FUSILLI_RETURN_ERROR_IF(getName().empty(), ErrorCode::AttributeNotSet,
                            "Graph name not set");
    // Validate nodes:
    // This infers missing tensor properties such as dims,
    // stride, dtype based on context.
    FUSILLI_CHECK_ERROR(validateSubtree());
    // Validate inputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on inputs first.
    for (const auto &input : fullGraphInputs_) {
      FUSILLI_CHECK_ERROR(input->validate());
    }
    // Validate outputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on outputs first.
    for (const auto &output : fullGraphOutputs_) {
      FUSILLI_CHECK_ERROR(output->validate());
    }
    FUSILLI_LOG_LABEL_ENDL("INFO: Graph validation completed successfully");
    isValidated_ = true;
    return ok();
  }

  // Compiles the graph using IREE compiler and sets up the IREE runtime
  // session context for future g->execute calls.
  //
  // Set `remove = true` to remove compilation artifacts (cache files) when
  // this `Graph` instance goes out of scope.
  ErrorObject compile(const Handle &handle, bool remove = false) {
    FUSILLI_RETURN_ERROR_IF(!isValidated_, ErrorCode::NotValidated,
                            "Graph must be validated before being compiled");
    return compileImpl(handle, remove);
  }

  // Executes the graph using IREE runtime. Requires a `variantPack` which is a
  // map from `TensorAttr` to `Buffer` wrapping the `iree_hal_buffer_view_t *`.
  // Definition in `fusilli/backend/runtime.h`.
  //
  // Backend Specific Execution Behavior
  //   For some backends execution will be async. The specifics of how one
  //   should launch a kernel and synchronize work items vary per backend.
  //
  // CPU
  //   Synchronous execution, no synchronization necessary.
  //
  // AMDGPU
  //   Asynchronous execution. Synchronization will depend on if you're using an
  //   external hip stream.

  //   `Handle::create(Backend::AMDGPU)`
  //      When not using an external hip stream, kernel launches are async and
  //      launched on the default (null) stream. Reads, writes, and allocations
  //      done through the fusilli APIs (`fusilli::Buffer::allocate`,
  //      `fusilli::Buffer::read`, etc.) are stream ordered (aka happen in the
  //      order executed). Read implicitly synchronizes (waits for anything in
  //      the stream behind it to finish) ensuring correct data is read. Any GPU
  //      read or write done outside of the fusilli APIs would lead to undefined
  //      behavior.
  //
  //      Note: The default stream has implicit synchronization with all other
  //      streams and will therefore limit concurrency with other streams.
  //
  //   `Handle::create(Backend::AMDGPU, /*deviceID=*/0, /*stream=*/stream)`
  //      With an external hip stream all kernel launches will be async and
  //      stream ordered on the stream provided. Assuming the default stream
  //      isn't used, there will be no synchronization with other streams. Any
  //      stream interaction will maintain normal stream ordering:
  //      `hipMallocAsync` (`hipMalloc` is synchronous so by default safe),
  //      `hipMemcpyAsync`, etc. are all fine.  Fusilli APIs are (still) stream
  //      ordered, and `fusilli::Buffer::read` maintains the synchronization
  //      behavior from the previous case.
  //
  //   Example Usage:
  //
  //     // Default stream  ===
  //     auto handle = Handle::create(Backend::AMDGPU);
  //     Graph graph;
  //
  //     // Allocate outputs
  //     auto outputBuf = std::make_shared<Buffer>(Buffer::allocate(...));
  //
  //     // Execute (async, stream ordered)
  //     graph.execute(handle, {{outputAttr, outputBuf}, ...});
  //
  //     // Read output (implicitly synchronizes stream)
  //     std::vector<half> result;
  //     outputBuf->read(handle, result);
  //
  //     // External stream  ===
  //     hipStream_t stream;
  //     hipStreamCreate(&stream);
  //     auto handle = Handle::create(Backend::AMDGPU, /*deviceID=*/0,
  //                                    /*stream=*/stream);
  //
  //     void *devicePtr;
  //     hipMallocAsync(&devicePtr, bufferSize, stream));
  //     ....
  //     auto outputBuf = std::make_shared<Buffer>(
  //                         Buffer::import(ireeBufferViewFromDevicePtr));
  //
  //     // All operations are stream ordered on the provided stream
  //     auto outputBuf = std::make_shared<Buffer>(Buffer::allocate(...));
  //     graph.execute(handle, {{outputAttr, outputBuf}, ...});
  //
  //     // Can mix with HIP operations on same stream
  //     hipMemcpyAsync(hostData, devicePtr, size,
  //                     hipMemcpyDeviceToDevice, stream);
  //     hipStreamSynchronize(stream);
  //     doSomethingWith(hostData);
  //
  // Workspace Buffer Usage:
  //   After calling compile(), query getWorkspaceSize() to determine if a
  //   workspace buffer is needed. If size > 0, allocate using
  //   Buffer::allocateRaw() and pass it to execute(). The same workspace
  //   buffer can be reused across multiple execute() calls.
  //
  //   Example:
  //     graph.compile(handle);
  //     auto wsSize = graph.getWorkspaceSize();
  //     std::shared_ptr<Buffer> workspace = nullptr;
  //     if (wsSize.value_or(0) > 0) {
  //       FUSILLI_ASSIGN_OR_RETURN(auto wsBuf,
  //                                Buffer::allocateRaw(handle, *wsSize));
  //       workspace = std::make_shared<Buffer>(std::move(wsBuf));
  //     }
  //     graph.execute(handle, variantPack, workspace);
  ErrorObject
  execute(const Handle &handle,
          const std::unordered_map<std::shared_ptr<TensorAttr>,
                                   std::shared_ptr<Buffer>> &variantPack,
          const std::shared_ptr<Buffer> &workspace = nullptr) const {
    return executeImpl(handle, variantPack, workspace);
  }

  // Delete copy constructors, keep default move constructor and destructor.
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;
  ~Graph() = default;

  // Getters and setters for graph context.
  const std::string &getName() const override final {
    return context.getName();
  }
  Type getType() const override final { return Type::Composite; }

  Graph &setName(const std::string &name) {
    context.setName(name);
    return *this;
  }

  Graph &setIODataType(DataType type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &setComputeDataType(DataType type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &setIntermediateDataType(DataType type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  // Declarations for tensor and op builder methods go here.
  // Definitions are towards the end of this file below.
  std::shared_ptr<TensorAttr> tensor(const TensorAttr &tensor);

  std::shared_ptr<TensorAttr> convFProp(const std::shared_ptr<TensorAttr> &x,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvFPropAttr &attributes);
  std::shared_ptr<TensorAttr> convWGrad(const std::shared_ptr<TensorAttr> &dy,
                                        const std::shared_ptr<TensorAttr> &x,
                                        ConvWGradAttr &attributes);
  std::shared_ptr<TensorAttr> convDGrad(const std::shared_ptr<TensorAttr> &dy,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvDGradAttr &attributes);
  std::array<std::shared_ptr<TensorAttr>, 3>
  layernorm(const std::shared_ptr<TensorAttr> &x,
            const std::shared_ptr<TensorAttr> &scale,
            const std::shared_ptr<TensorAttr> &bias, LayernormAttr &attributes);
  std::shared_ptr<TensorAttr> matmul(const std::shared_ptr<TensorAttr> &a,
                                     const std::shared_ptr<TensorAttr> &b,
                                     MatmulAttr &attributes);
  std::shared_ptr<TensorAttr> pointwise(const std::shared_ptr<TensorAttr> &in,
                                        PointwiseAttr &attributes);

  std::shared_ptr<TensorAttr> pointwise(const std::shared_ptr<TensorAttr> &in0,
                                        const std::shared_ptr<TensorAttr> &in1,
                                        PointwiseAttr &attributes);

  std::shared_ptr<TensorAttr> reduction(const std::shared_ptr<TensorAttr> &x,
                                        ReductionAttr &attributes);

  // ASM emitter driver method.
  //
  // TODO(#13): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::string> emitAsm() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Emitting MLIR assembly for Graph");
    FUSILLI_RETURN_ERROR_IF(
        !isValidated_, ErrorCode::NotValidated,
        "Graph must be validated before emitting MLIR assembly");
    std::ostringstream oss;
    emitAsmSubtree(oss);
    FUSILLI_LOG_ENDL(oss.str());
    return ok(oss.str());
  }

  // CRTP interface for GraphCRTP<Graph>.
  const std::string &getGraphName() const { return getName(); }

  ErrorOr<std::string> getAsm() { return emitAsm(); }

  // Populates IREE runtime call inputs from the variantPack for Graph.
  // Outputs first (skipping virtual), then inputs (skipping scalar).
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject populateCallInputs(
      iree_runtime_call_t &call,
      const std::unordered_map<std::shared_ptr<TensorAttr>,
                               std::shared_ptr<Buffer>> &variantPack) const;

private:
  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Adding output tensor '"
                           << name << "' to Graph outputs");
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating Graph");
    // Validate input/output names are unique (requirement for SSA).
    std::unordered_set<std::string> usedSymbols;
    for (const auto &t : fullGraphInputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    for (const auto &t : fullGraphOutputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    // Recursively validate node names are unique (requirement for SSA).
    FUSILLI_CHECK_ERROR(checkNodeNamesAreUnique(usedSymbols));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for Graph");
    // Populate sorted inputs / outputs after graph is fully constructed
    // and pre-validated (to ensure no symbol conflict).
    fullGraphInputsSorted_.insert(fullGraphInputs_.begin(),
                                  fullGraphInputs_.end());
    fullGraphOutputsSorted_.insert(fullGraphOutputs_.begin(),
                                   fullGraphOutputs_.end());
    return ok();
  }

  ErrorObject postValidateNode() const override final { return ok(); }

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string emitNodePostAsm() const override final;
  std::string getOperandNamesAndTypesAsm() const;
  std::string getResultNamesAndTypesAsm() const;

  // This is set after `validate()` is run at least once successfully.
  bool isValidated_ = false;

  // This is safe for post-insertion updates of TensorAttr (e.g. setting name
  // or other properties) since it uses the pointer value itself for hashing.
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphInputs_;
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphOutputs_;

  // These are sorted by the TensorAttr name, so post-insertion modification is
  // UB (undefined behavior). These are to be populated after the graph is fully
  // constructed and validated, and no further updates are expected.
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphInputsSorted_;
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphOutputsSorted_;
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(const TensorAttr &tensor) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Adding input tensor '" << tensor.getName()
                                                       << "' to Graph inputs");
  auto tensorPtr = std::make_shared<TensorAttr>(tensor);
  fullGraphInputs_.insert(tensorPtr);
  return tensorPtr;
}

// Create a ConvFPropNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convFProp(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvFPropAttr &convAttr) {
  // Populate names when not set.
  if (convAttr.getName().empty())
    convAttr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(convAttr.getName() + "_X");
  if (w && w->getName().empty())
    w->setName(convAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvFPropNode '" << convAttr.getName()
                                                        << "' to Graph");

  // Set inputs.
  convAttr.setX(x).setW(w);

  // Set outputs.
  auto y = outputTensor(convAttr.getName() + "_Y");
  convAttr.setY(y);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(convAttr), context));

  return y;
}

// Create a ConvWGradNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convWGrad(const std::shared_ptr<TensorAttr> &dy,
                 const std::shared_ptr<TensorAttr> &x,
                 ConvWGradAttr &convWGradAttr) {
  // Populate names when not set.
  if (convWGradAttr.getName().empty())
    convWGradAttr.setName("conv_wgrad_" + std::to_string(subNodes_.size()));
  if (dy && dy->getName().empty())
    dy->setName(convWGradAttr.getName() + "_DY");
  if (x && x->getName().empty())
    x->setName(convWGradAttr.getName() + "_X");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvWGradNode '"
                         << convWGradAttr.getName() << "' to Graph");

  // Set inputs.
  convWGradAttr.setDY(dy).setX(x);

  // Set outputs.
  auto dw = outputTensor(convWGradAttr.getName() + "_DW");
  convWGradAttr.setDW(dw);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvWGradNode>(std::move(convWGradAttr), context));

  return dw;
}

// Create a ConvDGradNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convDGrad(const std::shared_ptr<TensorAttr> &dy,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvDGradAttr &convDGradAttr) {
  // Populate names when not set.
  if (convDGradAttr.getName().empty())
    convDGradAttr.setName("conv_dgrad_" + std::to_string(subNodes_.size()));
  if (dy && dy->getName().empty())
    dy->setName(convDGradAttr.getName() + "_DY");
  if (w && w->getName().empty())
    w->setName(convDGradAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvDGradNode '"
                         << convDGradAttr.getName() << "' to Graph");

  // Set inputs.
  convDGradAttr.setDY(dy).setW(w);

  // Set outputs.
  auto dx = outputTensor(convDGradAttr.getName() + "_DX");
  convDGradAttr.setDX(dx);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvDGradNode>(std::move(convDGradAttr), context));

  return dx;
}

// Create a LayerNormNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes
inline std::array<std::shared_ptr<TensorAttr>, 3>
Graph::layernorm(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &scale,
                 const std::shared_ptr<TensorAttr> &bias,
                 LayernormAttr &layernormAttr) {
  // Populate names when not set.
  if (layernormAttr.getName().empty())
    layernormAttr.setName("layernorm_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(layernormAttr.getName() + "_X");
  if (scale && scale->getName().empty())
    scale->setName(layernormAttr.getName() + "_SCALE");
  if (bias && bias->getName().empty())
    bias->setName(layernormAttr.getName() + "_BIAS");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding LayerNorm '" << layernormAttr.getName()
                                                    << "' to Graph");

  // Set inputs.
  layernormAttr.setX(x);
  layernormAttr.setSCALE(scale);
  layernormAttr.setBIAS(bias);

  // Set outputs.
  std::shared_ptr<TensorAttr> y = outputTensor(layernormAttr.getName() + "_Y");
  std::shared_ptr<TensorAttr> m = nullptr;
  std::shared_ptr<TensorAttr> v = nullptr;
  if (layernormAttr.getForwardPhase() == NormFwdPhase::TRAINING) {
    m = outputTensor(layernormAttr.getName() + "_MEAN");
    v = outputTensor(layernormAttr.getName() + "_INV_VARIANCE");
  }
  layernormAttr.setY(y);
  layernormAttr.setMEAN(m);
  layernormAttr.setINV_VARIANCE(v);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<LayerNormNode>(std::move(layernormAttr), context));

  // `std::move` is useful for this case because we're returning an
  // array initialized from lvalues and `std::move` avoids unnecessary
  // copy and ref count operations on the shared pointers. This isn't
  // necessary in methods where a single local variable is returned
  // as NRVO would handle it.
  return {std::move(y), std::move(m), std::move(v)};
}

// Create a MatmulNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::matmul(const std::shared_ptr<TensorAttr> &a,
              const std::shared_ptr<TensorAttr> &b, MatmulAttr &matmulAttr) {
  // Populate names when not set.
  if (matmulAttr.getName().empty())
    matmulAttr.setName("matmul_" + std::to_string(subNodes_.size()));
  if (a && a->getName().empty())
    a->setName(matmulAttr.getName() + "_A");
  if (b && b->getName().empty())
    b->setName(matmulAttr.getName() + "_B");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding MatmulNode '" << matmulAttr.getName()
                                                     << "' to Graph");

  // Set inputs.
  matmulAttr.setA(a).setB(b);

  // Set outputs.
  auto c = outputTensor(matmulAttr.getName() + "_C");
  matmulAttr.setC(c);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<MatmulNode>(std::move(matmulAttr), context));

  return c;
}

// Create a PointwiseNode for single operand cases (e.g. RELU), populate it with
// the specified attributes, create output tensors and add the node to the
// graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::pointwise(const std::shared_ptr<TensorAttr> &in,
                 PointwiseAttr &pointwiseAttr) {
  // Populate names when not set.
  if (pointwiseAttr.getName().empty())
    pointwiseAttr.setName("pointwise_" + std::to_string(subNodes_.size()));
  if (in && in->getName().empty())
    in->setName(pointwiseAttr.getName() + "_IN_0");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding PointwiseNode '"
                         << pointwiseAttr.getName() << "' to Graph");

  // Set inputs.
  pointwiseAttr.setIN_0(in);

  // Set outputs.
  auto out = outputTensor(pointwiseAttr.getName() + "_OUT_0");
  pointwiseAttr.setOUT_0(out);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<PointwiseNode>(std::move(pointwiseAttr), context));

  return out;
}

// Create a PointwiseNode for cases with two operands (e.g. ADD), populate it
// with the specified attributes, create output tensors and add the node to the
// graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::pointwise(const std::shared_ptr<TensorAttr> &in0,
                 const std::shared_ptr<TensorAttr> &in1,
                 PointwiseAttr &pointwiseAttr) {
  // Populate names when not set.
  if (pointwiseAttr.getName().empty())
    pointwiseAttr.setName("pointwise_" + std::to_string(subNodes_.size()));
  if (in0 && in0->getName().empty())
    in0->setName(pointwiseAttr.getName() + "_IN_0");
  if (in1 && in1->getName().empty())
    in1->setName(pointwiseAttr.getName() + "_IN_1");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding PointwiseNode '"
                         << pointwiseAttr.getName() << "' to Graph");

  // Set inputs.
  pointwiseAttr.setIN_0(in0).setIN_1(in1);

  // Set outputs.
  auto out = outputTensor(pointwiseAttr.getName() + "_OUT_0");
  pointwiseAttr.setOUT_0(out);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<PointwiseNode>(std::move(pointwiseAttr), context));

  return out;
}

// Create a ReductionNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::reduction(const std::shared_ptr<TensorAttr> &x,
                 ReductionAttr &reductionAttr) {
  // Populate names when not set.
  if (reductionAttr.getName().empty())
    reductionAttr.setName("reduction_" + std::to_string(subNodes_.size()));
  if (x && x->getName().empty())
    x->setName(reductionAttr.getName() + "_X");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ReductionNode '"
                         << reductionAttr.getName() << "' to Graph");

  // Set inputs.
  reductionAttr.setX(x);

  // Set outputs.
  auto y = outputTensor(reductionAttr.getName() + "_Y");
  reductionAttr.setY(y);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ReductionNode>(std::move(reductionAttr), context));

  return y;
}

} // namespace fusilli

#endif // FUSILLI_GRAPH_GRAPH_H
