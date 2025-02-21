#pragma once
#include "runtime/performanceInfo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <regex>
using namespace tensorFly::runtime;
// Function to clean the value by removing leading and trailing spaces
std::string cleanValue(const std::string& value) {
    size_t first = value.find_first_not_of(" \t");
    size_t last = value.find_last_not_of(" \t");
    return value.substr(first, (last - first + 1));
}

// Function to parse the report from a string
void performanceInfo::parseReport(const std::string& reportContent) {
    std::map<std::string, std::string> metrics;

    // Regular expressions for matching the metric and its value
    std::regex metricPattern(R"((\w[\w\s/]+)\s+(\w+)\s+(\S+))");
    std::smatch match;

    // Process each line from the input string (split by newline)
    std::istringstream stream(reportContent);
    std::string line;

    while (std::getline(stream, line)) {
        // Look for lines with metric names and values
        if (std::regex_search(line, match, metricPattern)) {
            std::string metricName = cleanValue(match[1].str());
            std::string metricUnit = cleanValue(match[2].str());
            std::string metricValue = cleanValue(match[3].str());

            // Store the metric in the map
            metrics[metricName] = metricValue + " " + metricUnit;
        }
    }

    // Print the extracted metrics
    std::cout << "Extracted Metrics:" << std::endl;
    for (const auto& [metricName, value] : metrics) {
        std::cout << metricName << ": " << value << std::endl;
    }
}