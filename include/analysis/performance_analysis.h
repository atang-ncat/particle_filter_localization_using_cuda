#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct BenchmarkResult {
    int particle_count;
    double predict_time;
    double update_time;
    double resample_time;
    double total_time;
};

class PerformanceAnalyzer {
public:
    // Load benchmark results from file
    std::vector<BenchmarkResult> loadResults(const std::string& filename);
    
    // Generate scaling analysis visualization
    void createScalingAnalysisPlot(const std::vector<BenchmarkResult>& results, 
                                  const std::string& output_filename);
    
    // Generate component breakdown visualization
    void createComponentBreakdownPlot(const std::vector<BenchmarkResult>& results,
                                     const std::string& output_filename);
    
};