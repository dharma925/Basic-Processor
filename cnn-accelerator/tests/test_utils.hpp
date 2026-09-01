// Deliberately minimal, dependency-free test macros — no test framework to hide behind.
// Each CHECK just prints and counts failures; main() in each test file reports pass/fail.
#pragma once

#include <iostream>

inline int g_failures = 0;

#define CHECK(cond)                                                                    \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::cerr << "CHECK FAILED: " << #cond << " at " << __FILE__ << ":"         \
                       << __LINE__ << std::endl;                                        \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                                  \
    do {                                                                                \
        auto _a = (a);                                                                  \
        auto _b = (b);                                                                  \
        if (!(_a == _b)) {                                                              \
            std::cerr << "CHECK_EQ FAILED: " << #a << " == " << #b << "  (" << _a       \
                       << " != " << _b << ") at " << __FILE__ << ":" << __LINE__        \
                       << std::endl;                                                    \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

#define CHECK_THROWS(expr)                                                              \
    do {                                                                                \
        bool _threw = false;                                                            \
        try {                                                                           \
            (expr);                                                                     \
        } catch (...) {                                                                 \
            _threw = true;                                                              \
        }                                                                               \
        if (!_threw) {                                                                  \
            std::cerr << "CHECK_THROWS FAILED: " << #expr << " did not throw at "       \
                       << __FILE__ << ":" << __LINE__ << std::endl;                     \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)
