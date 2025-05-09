#include "particle_filter/particle_filter.cuh"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <random>

int main() {
    // Initialize particle filter with only num_particles parameter
    int num_particles = 500;
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

    for (int step = 0; step < 100; step++) {
        // Generate control inputs
        float velocity = 1.0f;
        float steering_angle = 0.1f * sinf(step * 0.05f);

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

        // Predict particle positions (assuming this method exists)
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

            // Create a measurement with the correct format for your implementation
            measurements.push_back(Measurement{
                static_cast<float>(i),  // id
                range,                // range
                static_cast<int>(bearing)               // bearing
            });
        }

        // Update weights with measurements and landmark locations
        // Based on error message, updateWeights takes 3 parameters
        pf.updateWeights(measurements, landmarks_x, landmarks_y);

        // Resample particles
        pf.resample();

        // Visualize (assuming this method exists)
        image.setTo(cv::Scalar(255, 255, 255));

        // Draw landmarks
        for (size_t i = 0; i < landmarks_x.size(); i++) {
            int lx = static_cast<int>(landmarks_x[i] * scale);
            int ly = static_cast<int>(landmarks_y[i] * scale);
            cv::circle(image, cv::Point(lx, ly), 8, cv::Scalar(0, 0, 0), -1);
            cv::putText(image, "L" + std::to_string(i), cv::Point(lx+10, ly),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        // Draw true position
        int tx = static_cast<int>(x * scale);
        int ty = static_cast<int>(y * scale);
        cv::circle(image, cv::Point(tx, ty), 5, cv::Scalar(255, 0, 0), -1);
        cv::line(image, cv::Point(tx, ty),
                cv::Point(tx + 15*cosf(theta), ty + 15*sinf(theta)),
                cv::Scalar(255, 0, 0), 2);

        // Draw particles (assuming this method exists)
        pf.visualize(image, scale);

        cv::imshow("Particle Filter", image);
        if (cv::waitKey(50) == 27) break;  // ESC to exit
    }

    cv::destroyAllWindows();
    return 0;
}