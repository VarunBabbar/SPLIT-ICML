//
// Created by ilias on 04/05/23.
//

#ifndef GOSDT_LOGGING_H
#define GOSDT_LOGGING_H

#include <cstdlib>
#include <iostream>

template <class... Args>
[[noreturn]] void gosdt_error(Args... args) {
    std::cout << "[GOSDT] ";
    (std::cout << ... << args);
    exit(-1);
}

template <class... Args>
void gosdt_log(Args... args) {
    (std::cout << ... << args) << '\n';
}

template <class... Args>
void gosdt_verbose_log(bool verbose, Args... args) {
    if (verbose) {
        (std::cout << ... << args) << '\n';
    }
}

#define todo(...)                                                                                          \
    error("[TODO]: file name: ", __FILE_NAME__, ", function: ", __FUNCTION__, ", line number: ", __LINE__, \
          "\nContext: ", __VA_ARGS__, "\n")

template <class... Args>
void assert_log(bool assertion, Args... args) {
    if (!assertion) {
        (std::cout << ... << args) << '\n';
        exit(EXIT_FAILURE);
    }
}

#endif  // GOSDT_LOGGING_H
