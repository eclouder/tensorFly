#pragma once
#include "runtime/nsight_runtime.h"
#include "cstdlib"
#include <iostream>
#include <cstdio>

using namespace tensorFly;

std::string runtime::nsight_compute_runtime::run(const std::string path) {
    std::string cmd = "ncu " + path;
#ifdef _WIN32
    // Use _popen on Windows
    FILE* fp = _popen(cmd.c_str(), "r");
#else
    // Use popen on Linux/Mac
    FILE* fp = popen(cmd.c_str(), "r");
#endif

    if (fp == nullptr) {
        std::cerr << "ncu error" << std::endl;
    }
    std::string content;

    char buffer[128];
    while (std::fgets(buffer, sizeof(buffer), fp) != nullptr) {
        content += buffer; // Append the line to the string
    }

    int result = _pclose(fp);
    if (result == 0) {
    } else {
        std::cout << "ncu error" << result << std::endl;
    };
    return content;
}
