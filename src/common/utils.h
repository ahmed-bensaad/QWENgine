#pragma once
/// @file utils.h
/// Logging and CUDA error checking shared by every component.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace q38rt {

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

inline void log(LogLevel level, const char* file, int line, const char* fmt, ...) {
    const char* level_str = "DEBUG";
    if (level == LogLevel::INFO) level_str = "INFO";
    else if (level == LogLevel::WARNING) level_str = "WARN";
    else if (level == LogLevel::ERROR) level_str = "ERROR";
    fprintf(stderr, "[%s] %s:%d: ", level_str, file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define LOG_DEBUG(...) q38rt::log(q38rt::LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...) q38rt::log(q38rt::LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...) q38rt::log(q38rt::LogLevel::WARNING, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) q38rt::log(q38rt::LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

/// Any CUDA runtime error is fatal: a failed call would otherwise leave stale buffers and silently wrong results.
inline void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error %d: %s at %s:%d", err, cudaGetErrorString(err), file, line);
        exit(1);
    }
}

#define CHECK_CUDA(err) q38rt::check_cuda_error(err, __FILE__, __LINE__)

} // namespace q38rt
