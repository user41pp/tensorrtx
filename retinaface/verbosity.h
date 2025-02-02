#ifndef VERBOSITY_H
#define VERBOSITY_H

enum VerbosityLevel {
    SILENT = 0,
    ERROR = 1, 
    WARNING = 2,
    INFO = 3,
    DEBUG = 4,
    TRACE = 5
};

extern VerbosityLevel g_verbosity;

// Host-only logging
#define LOG(level, ...) \
    if (level <= g_verbosity) { \
        printf(__VA_ARGS__); \
    }

// Device-side logging (for use in CUDA kernels)
#define CUDA_LOG(verbosity_level, level, ...) \
    if (threadIdx.x == 0 && blockIdx.x == 0 && level <= verbosity_level) { \
        printf(__VA_ARGS__); \
    }

#endif // VERBOSITY_H 