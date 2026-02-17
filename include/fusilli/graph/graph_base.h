// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the `GraphCRTP<Derived>` base class that provides
// compile/execute/cache infrastructure shared between `Graph` (node-tree-based
// MLIR generation) and `CustomGraph` (pre-written MLIR).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_GRAPH_BASE_H
#define FUSILLI_GRAPH_GRAPH_BASE_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/compile_command.h"
#include "fusilli/backend/compile_session.h"
#include "fusilli/backend/handle.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/logging.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#define IREE_COMPILE_INPUT_FILENAME "iree-compile-input.mlir"
#define IREE_COMPILE_OUTPUT_FILENAME "iree-compile-output.vmfb"
#define IREE_COMPILE_COMMAND_FILENAME "iree-compile-command.txt"
#define IREE_COMPILE_STATISTICS_FILENAME "iree-compile-statistics.json"

namespace fusilli {

inline bool checkCompileBackendEnv() {
  const char *backend = std::getenv("FUSILLI_COMPILE_BACKEND_USE_CLI");
  return backend && strcmp(backend, "0") != 0;
}

template <typename Derived> class GraphCRTP {
public:
  GraphCRTP(const GraphCRTP &) = delete;
  GraphCRTP &operator=(const GraphCRTP &) = delete;
  GraphCRTP(GraphCRTP &&) noexcept = default;
  GraphCRTP &operator=(GraphCRTP &&) noexcept = default;
  ~GraphCRTP() = default;

  // Query required workspace buffer size.
  // Returns std::nullopt if not compiled, 0 if no workspace needed,
  // or the required size in bytes.
  std::optional<size_t> getWorkspaceSize() const { return workspaceSize_; }

  ErrorOr<std::string> readCompilationCacheFile(CachedAssetsType type) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Getting cached assets path");
    FUSILLI_RETURN_ERROR_IF(!cache_.has_value(), ErrorCode::FileSystemFailure,
                            "Cache not populated yet");

    switch (type) {
    case CachedAssetsType::Input:
      return cache_->input.read();
    case CachedAssetsType::Command:
      return cache_->command.read();
    case CachedAssetsType::Output:
      return cache_->output.read();
    case CachedAssetsType::Statistics:
      return cache_->statistics.read();
    default:
      return error(ErrorCode::InvalidAttribute, "Unknown CachedAssetsType");
    }
  }

  // Return compiled artifact. The first invocation will always generate
  // compiled artifact, subsequent invocations may return cached versions
  // assuming cache invalidation checks pass. Set `remove = true` to remove
  // cache files when this instance goes out of scope.
  //
  // `reCompiled` will be set to true if a value is passed and the cache was
  // (re)generated; this parameter is useful for testing.
  //
  // TODO(#13): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `compile()`.
  ErrorOr<std::filesystem::path>
  getCompiledArtifact(const Handle &handle, const std::string &generatedAsm,
                      bool remove, std::optional<bool> *reCompiled = nullptr) {
    // Check for cache hit.
    FUSILLI_ASSIGN_OR_RETURN(bool cacheValid,
                             validateCache(handle, generatedAsm));
    if (cacheValid) {
      if (reCompiled)
        *reCompiled = false;
      return ok(cache_->output.path);
    }
    // (Re)generate cache.
    FUSILLI_ASSIGN_OR_RETURN(
        auto generatedCache,
        generateCompiledArtifact(handle, generatedAsm, remove));
    cache_ = std::move(generatedCache);
    if (reCompiled)
      *reCompiled = true;
    return ok(cache_->output.path);
  }

protected:
  GraphCRTP() = default;

  Derived &self() { return static_cast<Derived &>(*this); }
  const Derived &self() const { return static_cast<const Derived &>(*this); }

  // Core compile — called by derived class's public compile().
  ErrorObject compileImpl(const Handle &handle, bool remove = false) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Compiling Graph");

    // Generate MLIR assembly.
    FUSILLI_ASSIGN_OR_RETURN(std::string generatedAsm, self().getAsm());

    // Compile using IREE compiler or reuse cached artifact.
    FUSILLI_ASSIGN_OR_RETURN(auto vmfbPath,
                             getCompiledArtifact(handle, generatedAsm, remove));

    FUSILLI_LOG_LABEL_ENDL("INFO: Compiled Graph cached at \"" +
                           vmfbPath.string() + "\"");

    // Create per-graph IREE runtime session and load the compiled artifact.
    FUSILLI_CHECK_ERROR(createPerGraphSession(handle, vmfbPath.string()));

    return ok();
  }

  // Core execute — called by derived class's public execute().
  // Definition in `fusilli/backend/runtime.h` (needs IREE runtime API types).
  //
  // `workspace` — transient storage buffer required by some compiled modules.
  // Query getWorkspaceSize() after compile() to determine if one is needed.
  //
  // `outputs` — if non-null, receives any buffer views returned by the IREE
  // function call. This is needed when the compiled function returns outputs
  // as return values (sync execution path) rather than writing into
  // pre-allocated output argument buffers (async execution path).
  ErrorObject
  executeImpl(const Handle &handle,
              const std::unordered_map<std::shared_ptr<TensorAttr>,
                                       std::shared_ptr<Buffer>> &variantPack,
              const std::shared_ptr<Buffer> &workspace = nullptr,
              std::vector<Buffer> *outputs = nullptr) const;

private:
  // Definition in `fusilli/backend/runtime.h` (needs IREE runtime API types).
  ErrorObject createPerGraphSession(const Handle &handle,
                                    const std::string &vmfbPath);

  // Queries the required transient/workspace buffer size from the compiled
  // module. Definition in `fusilli/backend/runtime.h`.
  ErrorOr<size_t> queryTransientSize();

  // Create compiled artifacts from graph writing results to the cache. Set
  // `remove = true` to remove cache files when returned `CachedAssets` lifetime
  // ends.
  ErrorOr<CachedAssets>
  generateCompiledArtifact(const Handle &handle,
                           const std::string &generatedAsm, bool remove) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Generating compiled artifacts");

    // Create cache files.
    FUSILLI_ASSIGN_OR_RETURN(auto inputCache,
                             CacheFile::create(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_INPUT_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto outputCache,
                             CacheFile::create(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto commandCache,
                             CacheFile::create(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_COMMAND_FILENAME,
                                 /*remove=*/remove));
    FUSILLI_ASSIGN_OR_RETURN(auto statisticsCache,
                             CacheFile::create(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME,
                                 /*remove=*/remove));
    CachedAssets cache = CachedAssets(
        /*in=*/std::move(inputCache),
        /*out=*/std::move(outputCache),
        /*cmd=*/std::move(commandCache),
        /*stats=*/std::move(statisticsCache));

    // Write input asm to cache.
    FUSILLI_CHECK_ERROR(cache.input.write(generatedAsm));

    // determine which implementation to use.
    if (checkCompileBackendEnv()) {
      // Use CompileCommand (CLI).
      CompileCommand cmd = CompileCommand::build(
          handle, cache.input, cache.output, cache.statistics);
      FUSILLI_CHECK_ERROR(cmd.writeTo(cache.command));
      FUSILLI_LOG_LABEL_ENDL("INFO: iree-compile command (CLI)");
      FUSILLI_LOG_ENDL(cmd.toString());
      FUSILLI_CHECK_ERROR(cmd.execute());
    } else {
      // Use CompileSession (C API) - DEFAULT.
      FUSILLI_ASSIGN_OR_RETURN(CompileSession session,
                               CompileSession::build(handle, cache.input,
                                                     cache.output,
                                                     cache.statistics));
      FUSILLI_CHECK_ERROR(session.writeTo(cache.command));
      FUSILLI_LOG_LABEL_ENDL("INFO: iree-compile command (C API)");
      FUSILLI_LOG_ENDL(session.toString());
      FUSILLI_CHECK_ERROR(session.execute());
    }

    return ok(std::move(cache));
  }

  // Check for cache validity. Cache should be invalidated if:
  //  - Cache has not been generated for this instance yet
  //  - Graph name (and therefore cache path) has changed
  //  - Generated assembly differs
  //  - Compile commands have changed
  //  - Handle/backend (and therefore compile command) has changed
  ErrorOr<bool> validateCache(const Handle &handle,
                              const std::string &generatedAsm) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating cache");

    // Check for cache miss if cache hasn't been generated.
    if (!cache_.has_value()) {
      FUSILLI_LOG_ENDL("Cache not previously populated.");
      return ok(false);
    }

    // Check for cache miss if paths don't match (e.g., if graph name changed).
    if (cache_->input.path != CacheFile::getPath(
                                  /*graphName=*/self().getGraphName(),
                                  /*fileName=*/IREE_COMPILE_INPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache input paths differ.");
      return ok(false);
    }
    if (cache_->output.path != CacheFile::getPath(
                                   /*graphName=*/self().getGraphName(),
                                   /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache output paths differ.");
      return ok(false);
    }
    if (cache_->command.path !=
        CacheFile::getPath(
            /*graphName=*/self().getGraphName(),
            /*fileName=*/IREE_COMPILE_COMMAND_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile command paths differ.");
      return ok(false);
    }
    if (cache_->statistics.path !=
        CacheFile::getPath(
            /*graphName=*/self().getGraphName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile statistics paths differ.");
      return ok(false);
    }

    // Open expected files.
    FUSILLI_ASSIGN_OR_RETURN(CacheFile input,
                             CacheFile::open(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_INPUT_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(CacheFile output,
                             CacheFile::open(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(CacheFile command,
                             CacheFile::open(
                                 /*graphName=*/self().getGraphName(),
                                 /*fileName=*/IREE_COMPILE_COMMAND_FILENAME));
    FUSILLI_ASSIGN_OR_RETURN(
        CacheFile statistics,
        CacheFile::open(
            /*graphName=*/self().getGraphName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME));

    // Check for a cache miss on generated assembly.
    FUSILLI_ASSIGN_OR_RETURN(std::string inputContents, input.read());
    if (inputContents != generatedAsm) {
      FUSILLI_LOG_ENDL("Generated assembly does not match");
      return ok(false);
    }

    // Check for a cache miss on compile command.
    std::string cmdString;

    if (checkCompileBackendEnv()) {
      // Use CompileCommand (CLI).
      CompileCommand cmd =
          CompileCommand::build(handle, input, output, statistics);
      cmdString = cmd.toString();
    } else {
      // Use CompileSession (C API) - DEFAULT.
      FUSILLI_ASSIGN_OR_RETURN(
          auto session,
          CompileSession::build(handle, input, output, statistics));
      cmdString = session.toString();
    }

    FUSILLI_ASSIGN_OR_RETURN(std::string commandContents, command.read());
    if (commandContents != cmdString) {
      FUSILLI_LOG_ENDL("Compile command does not match");
      return ok(false);
    }

    return ok(true);
  }

  // Required workspace buffer size in bytes. Set during createPerGraphSession()
  // by querying the iree.abi.transients.size.constant attribute.
  // std::nullopt indicates the graph has not been compiled yet.
  std::optional<size_t> workspaceSize_ = std::nullopt;

  // Whether the compiled module has a `module.main$async` entry point.
  // When true, both `@main$async` and `@main` accept a workspace `!hal.buffer`
  // argument (from --iree-torch-externalize-transients). Set during
  // queryTransientSize().
  bool hasAsyncEntryPoint_ = false;

  IreeRuntimeSessionUniquePtrType session_;
  std::optional<CachedAssets> cache_ = std::nullopt;
};

} // namespace fusilli

#endif // FUSILLI_GRAPH_GRAPH_BASE_H
