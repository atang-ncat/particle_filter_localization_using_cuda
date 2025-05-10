#include "analysis/performance_analysis.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <iostream>

std::vector<BenchmarkResult> PerformanceAnalyzer::loadResults(const std::string& filename) {
    std::vector<BenchmarkResult> results;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return results;
    }
    
    // Skip the first two lines (header)
    std::string line;
    std::getline(file, line); // Skip title
    std::getline(file, line); // Skip column headers
    
    // Read data lines
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        BenchmarkResult result;
        char comma;
        
        iss >> result.particle_count >> comma
            >> result.predict_time >> comma
            >> result.update_time >> comma
            >> result.resample_time >> comma
            >> result.total_time;
        
        if (iss) {
            results.push_back(result);
        }
    }
    
    return results;
}

void PerformanceAnalyzer::createScalingAnalysisPlot(const std::vector<BenchmarkResult>& results, 
                                                   const std::string& output_filename) {
    // Create a blank image for the plot (800x600)
    cv::Mat plot(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Define plot margins
    int margin_left = 80;
    int margin_bottom = 60;
    int plot_width = plot.cols - margin_left - 50;
    int plot_height = plot.rows - margin_bottom - 50;
    
    // Find min and max particle counts for x-axis
    double min_particles = results[0].particle_count;
    double max_particles = results.back().particle_count;
    
    // Find min and max times for y-axis
    double min_time = results[0].total_time;
    double max_time = 0.0;
    for (const auto& result : results) {
        max_time = std::max(max_time, result.total_time);
    }
    
    // Add some headroom to max values
    max_particles *= 1.2;
    max_time *= 1.2;
    
    // Draw axes
    cv::line(plot, cv::Point(margin_left, plot.rows - margin_bottom), 
                  cv::Point(margin_left + plot_width, plot.rows - margin_bottom), 
                  cv::Scalar(0, 0, 0), 2);
    cv::line(plot, cv::Point(margin_left, plot.rows - margin_bottom), 
                  cv::Point(margin_left, plot.rows - margin_bottom - plot_height), 
                  cv::Scalar(0, 0, 0), 2);
    
    // Draw points and connect them
    std::vector<cv::Point> points;
    for (const auto& result : results) {
        // Use log scale for x and y
        double x_ratio = log10(result.particle_count / min_particles) / log10(max_particles / min_particles);
        double y_ratio = log10(result.total_time / min_time) / log10(max_time / min_time);
        
        int x = margin_left + static_cast<int>(x_ratio * plot_width);
        int y = plot.rows - margin_bottom - static_cast<int>(y_ratio * plot_height);
        
        points.push_back(cv::Point(x, y));
        cv::circle(plot, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }
    
    // Connect points with lines
    for (size_t i = 1; i < points.size(); ++i) {
        cv::line(plot, points[i-1], points[i], cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw O(n) reference line
    std::vector<cv::Point> reference_points;
    for (size_t i = 0; i < results.size(); ++i) {
        double x_ratio = log10(results[i].particle_count / min_particles) / log10(max_particles / min_particles);
        // O(n) means time is proportional to particle count
        double expected_time = results[0].total_time * (results[i].particle_count / results[0].particle_count);
        double y_ratio = log10(expected_time / min_time) / log10(max_time / min_time);
        
        int x = margin_left + static_cast<int>(x_ratio * plot_width);
        int y = plot.rows - margin_bottom - static_cast<int>(y_ratio * plot_height);
        
        reference_points.push_back(cv::Point(x, y));
    }
    
    // Draw dashed O(n) line
    for (size_t i = 1; i < reference_points.size(); ++i) {
        cv::line(plot, reference_points[i-1], reference_points[i], cv::Scalar(0, 128, 0), 2, cv::LINE_AA, 0);
    }
    
    // Add axis labels and title
    cv::putText(plot, "Particle Count (log scale)", cv::Point(plot.cols/2 - 100, plot.rows - 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    cv::putText(plot, "Execution Time (ms)", cv::Point(margin_left - 70, 30), 
    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    cv::putText(plot, "log scale", cv::Point(margin_left - 40, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    cv::putText(plot, "GPU Particle Filter Scaling Analysis", cv::Point(plot.cols/2 - 150, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
    
    // Add legend
    cv::putText(plot, "Measured Time", cv::Point(plot.cols - 180, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    cv::line(plot, cv::Point(plot.cols - 200, 50), cv::Point(plot.cols - 190, 50), 
                  cv::Scalar(0, 0, 255), 2);
                  
    cv::putText(plot, "O(n) Reference", cv::Point(plot.cols - 180, 70), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 128, 0), 1);
    cv::line(plot, cv::Point(plot.cols - 200, 70), cv::Point(plot.cols - 190, 70), 
                  cv::Scalar(0, 128, 0), 2, cv::LINE_AA, 0);
    
    // X-axis tick labels
    for (const auto& result : results) {
        double x_ratio = log10(result.particle_count / min_particles) / log10(max_particles / min_particles);
        int x = margin_left + static_cast<int>(x_ratio * plot_width);
        
        cv::line(plot, cv::Point(x, plot.rows - margin_bottom), 
                     cv::Point(x, plot.rows - margin_bottom + 5), 
                     cv::Scalar(0, 0, 0), 1);
        
        cv::putText(plot, std::to_string(result.particle_count), 
                    cv::Point(x - 20, plot.rows - margin_bottom + 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Y-axis tick labels
    int num_y_ticks = 5;
    for (int i = 0; i <= num_y_ticks; i++) {
        // Calculate position in log scale
        double ratio = static_cast<double>(i) / num_y_ticks;
        double time_value = min_time * pow(max_time / min_time, ratio);
        int y = plot.rows - margin_bottom - static_cast<int>(ratio * plot_height);
        
        // Draw tick mark
        cv::line(plot, cv::Point(margin_left, y), 
                      cv::Point(margin_left - 5, y), 
                      cv::Scalar(0, 0, 0), 1);
        
        // Format and draw the label
        std::stringstream ss;
        ss << std::fixed << std::setprecision(time_value < 1.0 ? 3 : 1) << time_value;
        cv::putText(plot, ss.str(), 
                   cv::Point(margin_left - 45, y + 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }

    // Save the image
    cv::imwrite(output_filename, plot);
    std::cout << "Scaling analysis plot saved to " << output_filename << std::endl;
}

void PerformanceAnalyzer::createComponentBreakdownPlot(const std::vector<BenchmarkResult>& results,
                                                      const std::string& output_filename) {
    // Create a blank image for the plot (1000x600)
    cv::Mat plot(600, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Define plot margins
    int margin_left = 80;
    int margin_bottom = 60;
    int plot_width = plot.cols - margin_left - 50;
    int plot_height = plot.rows - margin_bottom - 50;
    
    // Bar parameters
    int num_bars = results.size();
    int bar_width = std::min(80, plot_width / (num_bars * 2));
    int bar_spacing = bar_width / 2;
    
    // Find max time for y-axis
    double max_time = 0.0;
    for (const auto& result : results) {
        max_time = std::max(max_time, result.total_time);
    }
    max_time *= 1.1; // Add 10% headroom
    
    // Draw axes
    cv::line(plot, cv::Point(margin_left, plot.rows - margin_bottom), 
                 cv::Point(margin_left + plot_width, plot.rows - margin_bottom), 
                 cv::Scalar(0, 0, 0), 2);
                 
    cv::line(plot, cv::Point(margin_left, plot.rows - margin_bottom), 
                 cv::Point(margin_left, plot.rows - margin_bottom - plot_height), 
                 cv::Scalar(0, 0, 0), 2);

    
    int num_y_ticks = 5;
    for (int i = 0; i <= num_y_ticks; i++) {
        // Calculate position in linear scale
        double ratio = static_cast<double>(i) / num_y_ticks;
        double time_value = ratio * max_time;
        int y = plot.rows - margin_bottom - static_cast<int>(ratio * plot_height);
        
        // Draw tick mark
        cv::line(plot, cv::Point(margin_left, y), 
                    cv::Point(margin_left - 5, y), 
                    cv::Scalar(0, 0, 0), 1);
        
        // Format and draw the label
        std::stringstream ss;
        ss << std::fixed << std::setprecision(time_value < 1.0 ? 3 : 1) << time_value;
        cv::putText(plot, ss.str(), 
                cv::Point(margin_left - 45, y + 5), 
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    
    // Define colors for components
    cv::Scalar predict_color(76, 175, 80);      // Green
    cv::Scalar update_color(33, 150, 243);      // Blue 
    cv::Scalar resample_color(244, 67, 54);     // Red
    
    // Draw stacked bars for each result
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // Calculate bar position
        int bar_center_x = margin_left + bar_width/2 + i * (bar_width + bar_spacing) * 2;
        int bar_bottom_y = plot.rows - margin_bottom;
        
        // Calculate component heights
        int predict_height = static_cast<int>((result.predict_time / max_time) * plot_height);
        int update_height = static_cast<int>((result.update_time / max_time) * plot_height);
        int resample_height = static_cast<int>((result.resample_time / max_time) * plot_height);
        
        // Calculate percentages for labels
        double predict_pct = 100.0 * result.predict_time / result.total_time;
        double update_pct = 100.0 * result.update_time / result.total_time;
        double resample_pct = 100.0 * result.resample_time / result.total_time;
        
        // Draw prediction component (bottom)
        cv::rectangle(plot, 
                     cv::Point(bar_center_x - bar_width/2, bar_bottom_y - predict_height), 
                     cv::Point(bar_center_x + bar_width/2, bar_bottom_y),
                     predict_color, -1);
        
        // Draw update component (middle)
        cv::rectangle(plot, 
                     cv::Point(bar_center_x - bar_width/2, bar_bottom_y - predict_height - update_height), 
                     cv::Point(bar_center_x + bar_width/2, bar_bottom_y - predict_height),
                     update_color, -1);
        
        // Draw resample component (top)
        cv::rectangle(plot, 
                     cv::Point(bar_center_x - bar_width/2, bar_bottom_y - predict_height - update_height - resample_height), 
                     cv::Point(bar_center_x + bar_width/2, bar_bottom_y - predict_height - update_height),
                     resample_color, -1);
        
        // Add percentage labels
        std::stringstream predict_label, update_label, resample_label;
        predict_label << std::fixed << std::setprecision(1) << predict_pct << "%";
        update_label << std::fixed << std::setprecision(1) << update_pct << "%";
        resample_label << std::fixed << std::setprecision(1) << resample_pct << "%";
        
        // Add percentage text in the middle of each segment
        if (predict_height > 15) {
            cv::putText(plot, predict_label.str(), 
                       cv::Point(bar_center_x - 20, bar_bottom_y - predict_height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
        
        if (update_height > 15) {
            cv::putText(plot, update_label.str(), 
                       cv::Point(bar_center_x - 20, bar_bottom_y - predict_height - update_height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
        
        if (resample_height > 15) {
            cv::putText(plot, resample_label.str(), 
                       cv::Point(bar_center_x - 20, bar_bottom_y - predict_height - update_height - resample_height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
        }
        
        // Add total execution time on top
        std::stringstream total_time;
        total_time << std::fixed << std::setprecision(2) << result.total_time << " ms";
        cv::putText(plot, total_time.str(),
                   cv::Point(bar_center_x - 25, bar_bottom_y - predict_height - update_height - resample_height - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        
        // Add x-axis labels (particle counts)
        cv::putText(plot, std::to_string(result.particle_count),
                   cv::Point(bar_center_x - 20, bar_bottom_y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    
    // Add title and axis labels
    cv::putText(plot, "Component-wise Breakdown of GPU Particle Filter Performance", 
               cv::Point(plot.cols/2 - 250, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
               
    cv::putText(plot, "Number of Particles", 
               cv::Point(plot.cols/2 - 70, plot.rows - 15),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
               
    cv::putText(plot, "Execution Time (ms)", 
                cv::Point(margin_left - 70, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
    
    // Add legend
    int legend_x = 750;
    int legend_y = 60;
    int legend_box_size = 15;
    
    cv::rectangle(plot, 
                 cv::Point(legend_x, legend_y), 
                 cv::Point(legend_x + legend_box_size, legend_y + legend_box_size),
                 predict_color, -1);
    cv::putText(plot, "Prediction", 
               cv::Point(legend_x + legend_box_size + 10, legend_y + legend_box_size),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    cv::rectangle(plot, 
                 cv::Point(legend_x, legend_y + 25), 
                 cv::Point(legend_x + legend_box_size, legend_y + 25 + legend_box_size),
                 update_color, -1);
    cv::putText(plot, "Weight Update", 
               cv::Point(legend_x + legend_box_size + 10, legend_y + 25 + legend_box_size),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    cv::rectangle(plot, 
                 cv::Point(legend_x, legend_y + 50), 
                 cv::Point(legend_x + legend_box_size, legend_y + 50 + legend_box_size),
                 resample_color, -1);
    cv::putText(plot, "Resampling", 
               cv::Point(legend_x + legend_box_size + 10, legend_y + 50 + legend_box_size),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    // Save the image
    cv::imwrite(output_filename, plot);
    std::cout << "Component breakdown plot saved to " << output_filename << std::endl;
}
