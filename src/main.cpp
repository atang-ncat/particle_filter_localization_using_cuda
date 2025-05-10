#include "particle_filter/particle_filter.cuh"
#include "analysis/performance_analysis.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h> 
#include <thrust/system/cuda/execution_policy.h>

// Simple timing function for benchmarking
template<typename Func>
double measure_time_ms(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Function to run benchmarks
void run_benchmarks() {
    // Parameters
    const int num_landmarks = 5;
    const int num_iterations = 100;
    const float dt = 0.1f;
    
    // Create landmarks
    std::vector<float> landmarks_x = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f};
    std::vector<float> landmarks_y = {5.0f, 15.0f, 25.0f, 10.0f, 20.0f};
    
    // Create measurements
    std::vector<Measurement> measurements;
    for (int i = 0; i < num_landmarks; i++) {
        measurements.push_back(Measurement{
            static_cast<float>(i),       // landmark_id
            5.0f + i,                    // range (simulated)
            static_cast<int>(0.1f * i)   // bearing (cast to int)
        });
    }
    
    // Test with different particle counts
    std::vector<int> particle_counts = {100, 1000, 10000, 100000};
    
    // Create and open a text file for writing results
    std::ofstream results_file("particle_filter_results.txt");
    
    // Write header to file
    results_file << "===== GPU Particle Filter Performance =====" << std::endl;
    results_file << "Particle Count, Prediction (ms), Weight Update (ms), Resampling (ms), Total (ms)" << std::endl;
    
    // Also output to console
    std::cout << "===== GPU Particle Filter Performance =====" << std::endl;
    std::cout << "\nParticle Count, Prediction (ms), Weight Update (ms), Resampling (ms), Total (ms)\n";
    
    for (int count : particle_counts) {
        // Initialize particle filter with this count
        ParticleFilter pf_gpu(count);
        
        // Time prediction step
        double predict_time_gpu = measure_time_ms([&]() {
            for (int i = 0; i < num_iterations; i++) {
                pf_gpu.predict(dt, 1.0f, 0.1f);
            }
        }) / num_iterations;
        
        // Time weight update step
        double update_weights_time_gpu = measure_time_ms([&]() {
            for (int i = 0; i < num_iterations; i++) {
                pf_gpu.updateWeights(measurements, landmarks_x, landmarks_y);
            }
        }) / num_iterations;
        
        // Time resampling step
        double resample_time_gpu = measure_time_ms([&]() {
            for (int i = 0; i < num_iterations; i++) {
                pf_gpu.resample();
            }
        }) / num_iterations;
        
        // Calculate total time
        double total_time = predict_time_gpu + update_weights_time_gpu + resample_time_gpu;
        
        // Format the result string
        std::stringstream result;
        result << count << ", " 
               << predict_time_gpu << ", " 
               << update_weights_time_gpu << ", " 
               << resample_time_gpu << ", "
               << total_time;
        
        // Output results to console
        std::cout << result.str() << std::endl;
        
        // Save to file
        results_file << result.str() << std::endl;
    }
    
    // Close the file
    results_file.close();
    
    std::cout << "\nBenchmark results saved to particle_filter_results.txt" << std::endl;
}


// Function to run the visualization simulation
void run_visualization() {
    // Initialize particle filter with only num_particles parameter
    int num_particles = 300;
    ParticleFilter pf(num_particles);

    // Define landmarks
    std::vector<float> landmarks_x = {5.0f, 10.0f, 15.0f, 20.0f, 25.0f};
    std::vector<float> landmarks_y = {5.0f, 15.0f, 25.0f, 10.0f, 20.0f};

    // Create visualization canvas
    cv::Mat image(600, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    float scale = 20.0f;  // pixels per meter

    // Random generator for simulating measurements
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.1);

    // Simulation parameters
    float x = 5.0f, y = 5.0f, theta = 0.0f;  // True position
    float dt = 0.1f;  // Time step
    float wheelbase = 1.0f;
    
    // Define a target position for the robot to reach
    float target_x = 20.0f, target_y = 20.0f;
    
    // Control parameters
    float steering_gain = 0.5f;  // How aggressively to steer toward goal
    float distance_threshold = 1.0f;  // How close to get to target

    for (int step = 0; step < 500; step++) {
        // Generate goal-directed control inputs
        float velocity = 1.0f;  // Keep constant velocity
        
        // Calculate angle to target
        float dx = target_x - x;
        float dy = target_y - y;
        float target_heading = atan2f(dy, dx);
        float heading_error = target_heading - theta;
        
        // Normalize heading error to [-pi, pi]
        while (heading_error > M_PI) heading_error -= 2.0f * M_PI;
        while (heading_error < -M_PI) heading_error += 2.0f * M_PI;
        
        // Calculate steering angle using proportional control
        float steering_angle = steering_gain * heading_error;
        
        // Check if we've reached the target
        float distance_to_target = sqrtf(dx*dx + dy*dy);
        if (distance_to_target < distance_threshold) {
            std::cout << "Target reached in " << step << " steps!" << std::endl;
            break;
        }

        // Update true position using bicycle model
        if (fabs(steering_angle) < 0.001f) {
            x += velocity * dt * cosf(theta);
            y += velocity * dt * sinf(theta);
        } else {
            float turning_radius = wheelbase / tanf(steering_angle);
            float angle_change = velocity / turning_radius * dt;
            x += turning_radius * (sinf(theta + angle_change) - sinf(theta));
            y += turning_radius * (cosf(theta) - cosf(theta + angle_change));
            theta += angle_change;
        }

        // Predict particle positions
        pf.predict(dt, velocity, steering_angle);

        // Generate measurements
        std::vector<Measurement> measurements;
        for (size_t i = 0; i < landmarks_x.size(); i++) {
            float dx = landmarks_x[i] - x;
            float dy = landmarks_y[i] - y;
            float range = sqrtf(dx*dx + dy*dy);
            float bearing = atan2f(dy, dx) - theta;

            // Add noise
            range += noise(gen);
            bearing += noise(gen) * 0.1f;

            // Normalize bearing
            while (bearing > M_PI) bearing -= 2.0f * M_PI;
            while (bearing < -M_PI) bearing += 2.0f * M_PI;

            measurements.push_back(Measurement{
                static_cast<float>(i),    // id
                range,                    // range
                static_cast<int>(bearing) // bearing needs to be int
            });
        }

        // Update weights with measurements and landmark locations
        pf.updateWeights(measurements, landmarks_x, landmarks_y);

        // Resample particles
        pf.resample();

        // Visualize
        image.setTo(cv::Scalar(255, 255, 255));

        // Draw landmarks
        for (size_t i = 0; i < landmarks_x.size(); i++) {
            int lx = static_cast<int>(landmarks_x[i] * scale);
            int ly = static_cast<int>(landmarks_y[i] * scale);
            cv::circle(image, cv::Point(lx, ly), 8, cv::Scalar(0, 0, 0), -1);
            cv::putText(image, "L" + std::to_string(i), cv::Point(lx+10, ly),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        // Draw target position
        int target_x_px = static_cast<int>(target_x * scale);
        int target_y_px = static_cast<int>(target_y * scale);
        cv::circle(image, cv::Point(target_x_px, target_y_px), 8, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, "TARGET", cv::Point(target_x_px+10, target_y_px),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

        // Draw true position
        int tx = static_cast<int>(x * scale);
        int ty = static_cast<int>(y * scale);
        cv::circle(image, cv::Point(tx, ty), 5, cv::Scalar(255, 0, 0), -1);
        cv::line(image, cv::Point(tx, ty),
                cv::Point(tx + 15*cosf(theta), ty + 15*sinf(theta)),
                cv::Scalar(255, 0, 0), 2);

        // Draw particles
        pf.visualize(image, scale);

        cv::imshow("Particle Filter", image);
        if (cv::waitKey(50) == 27) break;  // ESC to exit
    }

    cv::destroyAllWindows();
}

int main(int argc, char** argv) {
    // Check command line arguments to determine mode
    bool run_benchmark = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark" || arg == "-b") {
            run_benchmark = true;
        }
    }
    
    if (run_benchmark) {
        std::cout << "Running benchmarks..." << std::endl;
        run_benchmarks();

        PerformanceAnalyzer analyzer;
        std::vector<BenchmarkResult> results = analyzer.loadResults("particle_filter_results.txt");
        
        if (!results.empty()) {
            analyzer.createScalingAnalysisPlot(results, "scaling_analysis.png");
            analyzer.createComponentBreakdownPlot(results, "component_breakdown.png");

            std::cout << "Performance analysis complete. Check the output files." << std::endl;
        }
    } else {
        std::cout << "Running visualization..." << std::endl;
        run_visualization();
    }
    
    return 0;
}