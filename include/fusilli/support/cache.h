// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains classes for cache file handling of generated artifacts.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_CACHE_H
#define FUSILLI_SUPPORT_CACHE_H

#include "fusilli/support/logging.h"
#include "fusilli/support/target_platform.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <utility>

#if defined(FUSILLI_PLATFORM_WINDOWS)
#include <KnownFolders.h>
#include <shlobj.h>
#include <windows.h>
#endif

namespace fusilli {

// Generate a random 16-character hex string for per-instance cache isolation.
inline std::string generateCacheUid() {
  std::random_device rd;
  uint32_t hi = rd();
  uint32_t lo = rd();
  return std::format("{:08x}{:08x}", hi, lo);
}

// An RAII type for creating + destroying cache files in
// `${HOME}/.cache/fusilli`.
//
//  void example() {
//    std::string uid = generateCacheUid();
//
//    // `remove = true`
//    {
//      // Create ${HOME}/.cache/fusilli/example_graph/<uid>/input
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*uid=*/uid,
//          /*filename=*/"input", /*remove=*/true);
//      assert(isOk(cacheFile));
//
//      assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                  /*uid=*/uid, /*filename=*/"input")));
//    }
//    // Try to open the same (now removed) cache file.
//    assert(isError(CacheFile::open(/*graphName=*/"example_graph",
//                                   /*uid=*/uid, /*filename=*/"input")));
//
//    // `remove = false`
//    {
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*uid=*/uid,
//          /*filename=*/"input", /*remove=*/false);
//      assert(isOk(cacheFile));
//    }
//    // Try to open the same cache file. This time it's found.
//    assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                /*uid=*/uid, /*filename=*/"input")));
//  }
class CacheFile {
public:
  // Factory constructor that creates file, overwriting an existing file, and
  // returns an ErrorObject if file could not be created.
  static ErrorOr<CacheFile> create(const std::string &graphName,
                                   const std::string &uid,
                                   const std::string &fileName, bool remove) {
    std::filesystem::path path = CacheFile::getPath(graphName, uid, fileName);
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating Cache file");
    FUSILLI_LOG_ENDL(path);

    // Create directory: ${HOME}/.cache/fusilli/<graphName>/<uid>
    std::filesystem::path cacheDir = path.parent_path();
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    FUSILLI_RETURN_ERROR_IF(ec, ErrorCode::FileSystemFailure,
                            "Failed to create cache directory: " +
                                cacheDir.string() + " - " + ec.message());

    // Create file: ${HOME}/.cache/fusilli/<graphName>/<fileName>
    std::ofstream file(path);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to create file: " + path.string());
    file.close();

    return ok(CacheFile(path, remove));
  }

  // Factory constructor that opens an existing file and returns ErrorObject if
  // the file does not exist.
  static ErrorOr<CacheFile> open(const std::string &graphName,
                                 const std::string &uid,
                                 const std::string &fileName) {
    std::filesystem::path path = CacheFile::getPath(graphName, uid, fileName);

    // Check if the file exists.
    FUSILLI_RETURN_ERROR_IF(!std::filesystem::exists(path),
                            ErrorCode::FileSystemFailure,
                            "File does not exist: " + path.string());

    return ok(CacheFile(path, /*remove=*/false));
  }

  static std::filesystem::path getCacheDir() {
    // Defaults to "${HOME}/.cache/fusilli" but having it set via
    // ${FUSILLI_CACHE_DIR} to "/tmp" helps bypass permission issues on
    // the GitHub Actions CI runners as well as for LIT tests that rely
    // on dumping/reading intermediate compilation artifacts to/from disk.
    const char *cacheDirEnv = std::getenv("FUSILLI_CACHE_DIR");
    std::wstring cacheDir;
    if (cacheDirEnv) {
      std::string cacheDirStr(cacheDirEnv);
      cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
    }

#if defined(FUSILLI_PLATFORM_WINDOWS)
    if (cacheDir.empty()) {
      PWSTR pathBuf = nullptr;
      HRESULT hr =
          SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, NULL, &pathBuf);
      if (SUCCEEDED(hr)) {
        std::string cacheDirStr = std::filesystem::path(pathBuf).string();
        cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
      }
      if (pathBuf)
        CoTaskMemFree(pathBuf);
    }
    return std::filesystem::path(cacheDir) / "fusilli";
#else
    if (cacheDir.empty()) {
      const char *home = std::getenv("HOME");
      if (home) {
        std::string cacheDirStr(home);
        cacheDir = std::wstring(cacheDirStr.begin(), cacheDirStr.end());
      }
    }
    return std::filesystem::path(cacheDir) / ".cache" / "fusilli";
#endif
  }

  // Utility method to build the path to cache file given `graphName`, `uid`,
  // and `fileName`.
  //
  // Format: ${HOME}/.cache/fusilli/<sanitized graphName>/<uid>/<fileName>
  static std::filesystem::path getPath(const std::string &graphName,
                                       const std::string &uid,
                                       const std::string &fileName) {
    // Ensure graphName is safe to use as a directory name, we assume fileName
    // is safe.
    std::string sanitizedGraphName = graphName;
    std::transform(sanitizedGraphName.begin(), sanitizedGraphName.end(),
                   sanitizedGraphName.begin(),
                   [](char c) { return c == ' ' ? '_' : c; });
    std::erase_if(sanitizedGraphName, [](unsigned char c) { // C++20
      return !(std::isalnum(c) || c == '_');
    });

    // Ensure graphName has a value.
    if (sanitizedGraphName.empty())
      sanitizedGraphName = "unnamed_graph";

    std::filesystem::path cacheDir = getCacheDir();
    return cacheDir / sanitizedGraphName / uid / fileName;
  }

  // Move constructors.
  CacheFile(CacheFile &&other) noexcept
      : path(std::move(other.path)), remove_(other.remove_) {
    other.path.clear();
    other.remove_ = false;
  }
  CacheFile &operator=(CacheFile &&other) noexcept {
    if (this == &other)
      return *this;
    // If ownership of the cached file is simply changing, we aren't creating a
    // dangling resource that might to be removed.
    bool samePath = path == other.path;
    // Remove current resource if needed.
    if (remove_ && !path.empty() && !samePath)
      std::filesystem::remove(path);
    // Move from other.
    path = std::move(other.path);
    remove_ = other.remove_;
    other.path.clear();
    other.remove_ = false;
    return *this;
  }

  // Delete copy constructors. A copy constructor would likely not be safe, as
  // the destructor for a copy could remove the underlying file while the
  // original is still expecting it to exist.
  CacheFile(const CacheFile &) = delete;
  CacheFile &operator=(const CacheFile &) = delete;

  ~CacheFile() {
    if (remove_ && !path.empty()) {
      std::filesystem::remove(path);
    }
  }

  // Path of file this class wraps.
  std::filesystem::path path;

  // Write to cache file.
  ErrorObject write(const std::string &content) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to open file: " + path.string());

    file << content;
    FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                            "Failed to write to file: " + path.string());

    return ok();
  }

  // Read contents of cache file.
  ErrorOr<std::string> read() {
    // std::ios::ate opens file and moves the cursor to the end, allowing us
    // to get the file size with tellg().
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to open file: " + path.string());

    // Copy the contents of the file into a string.
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    file.read(buffer.data(), size);
    FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                            "Failed to read file: " + path.string());

    return ok(buffer);
  }

private:
  // Class should be constructed using one of the factory functions.
  CacheFile(std::filesystem::path path, bool remove)
      : path(std::move(path)), remove_(remove) {}

  // Whether to remove the file on destruction or not.
  bool remove_;
};

// CleanupCacheDirectory removes a sub-directory from the main cache directory
// if it's empty. When used as a base class, C++ destructor ordering (explained
// below) ensures that the directory cleanup in CleanupCacheDirectory destructor
// will happen after any CacheFiles member variables have been destroyed.
//
// Destructor ordering example:
//   struct A   { ~A()  {std::cout << "A";} };
//   struct M1  { ~M1() {std::cout << "M1, ";} };
//   struct M2  { ~M2() {std::cout << "M2, ";} };
//
//   struct B : A {
//       M1 m1;
//       M2 m2;
//       ~B() { std::cout << "B, "; }
//   };
//   // output -> "B, M2, M1, A"
//
// If member destructors (~M1, ~M2) are called inside ~B; the compiler will
// still destroy members afterward, leading to double-destruction (UB).
struct CleanupCacheDirectory {
  std::filesystem::path cacheDir;
  explicit CleanupCacheDirectory(std::filesystem::path dir)
      : cacheDir(std::move(dir)) {}

  CleanupCacheDirectory(CleanupCacheDirectory &&other) noexcept
      : cacheDir(std::move(other.cacheDir)) {}

  // Delete copy — directory ownership must transfer, not duplicate.
  CleanupCacheDirectory(const CleanupCacheDirectory &) = delete;
  CleanupCacheDirectory &operator=(const CleanupCacheDirectory &) = delete;
  CleanupCacheDirectory &operator=(CleanupCacheDirectory &&) = delete;

  ~CleanupCacheDirectory() {
    if (cacheDir.empty())
      return;
    removeEmptyDirAndParent(cacheDir);
  }

  // Remove `dir` if empty, then remove its parent if also empty. Uses
  // error_code overloads to avoid throwing in noexcept / destructor contexts.
  static void removeEmptyDirAndParent(const std::filesystem::path &dir) {
    std::error_code ec;
    if (std::filesystem::exists(dir, ec) && !ec &&
        std::filesystem::is_empty(dir, ec) && !ec)
      std::filesystem::remove(dir, ec);

    std::filesystem::path parent = dir.parent_path();
    if (!parent.empty() && std::filesystem::exists(parent, ec) && !ec &&
        std::filesystem::is_empty(parent, ec) && !ec)
      std::filesystem::remove(parent, ec);
  }
};

enum class CachedAssetsType : uint8_t {
  Input,
  Output,
  Command,
  Statistics,
};

// Holds cached assets. If `CacheFiles` are set to be removed RAII based removal
// will be tied to the lifetime of this object.
struct CachedAssets : CleanupCacheDirectory {
  CacheFile input;
  CacheFile output;
  CacheFile command;
  CacheFile statistics;

  CachedAssets(CacheFile &&in, CacheFile &&out, CacheFile &&cmd,
               CacheFile &&stats)
      : CleanupCacheDirectory(in.path.parent_path()), input(std::move(in)),
        output(std::move(out)), command(std::move(cmd)),
        statistics(std::move(stats)) {
    // sanity checks:
    assert(input.path.parent_path() == output.path.parent_path() &&
           input.path.parent_path() == command.path.parent_path() &&
           input.path.parent_path() == statistics.path.parent_path() &&
           "Cached assets should be in the same directory.");
    assert(std::filesystem::is_directory(input.path.parent_path()));
  }

  CachedAssets(CachedAssets &&) noexcept = default;
  ~CachedAssets() = default;

  // Custom move-assignment: move members FIRST so CacheFile::operator= removes
  // old files, then clean up the now-empty old directory before updating
  // cacheDir. The default operator= would process the base (cacheDir) before
  // members, leaking the old directory when the graph name changes.
  CachedAssets &operator=(CachedAssets &&other) noexcept {
    if (this == &other)
      return *this;

    std::filesystem::path oldDir = cacheDir;

    // Move members first — CacheFile::operator= removes old files when paths
    // differ (e.g., graph name changed).
    input = std::move(other.input);
    output = std::move(other.output);
    command = std::move(other.command);
    statistics = std::move(other.statistics);

    // Update cacheDir to the new directory.
    cacheDir = std::move(other.cacheDir);

    // Clean up old directory if it became empty and differs from the new one.
    if (!oldDir.empty() && oldDir != cacheDir)
      removeEmptyDirAndParent(oldDir);

    return *this;
  }

  // Delete copy constructors.
  CachedAssets(const CachedAssets &) = delete;
  CachedAssets &operator=(const CachedAssets &) = delete;
};

} // namespace fusilli

#endif // FUSILLI_SUPPORT_CACHE_H
