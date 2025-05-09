#include "particle_filter/measurement_model.cuh"
#include <math.h>

__global__ void updateWeightsKernel(
    float* x,
    float* y,
    float* theta,
    float* weights,
    float* landmarks_x,
    float* landmarks_y,
    Measurement* measurements,
    int num_measurements,
    float range_std,
    float bearing_std,
    int num_particles,
    int num_landmarks
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        float particle_x = x[idx];
        float particle_y = y[idx];
        float particle_theta = theta[idx];

        // Initialize weight
        float weight = 1.0f;

        // Process each measurement
        for (int m = 0; m < num_measurements; m++) {
            Measurement meas = measurements[m];
            int landmark_id = meas.landmark_id;

            // Skip if invalid landmark ID
            if (landmark_id < 0 || landmark_id >= num_landmarks) continue;

            // Get landmark position
            float landmark_x = landmarks_x[landmark_id];
            float landmark_y = landmarks_y[landmark_id];

            // Calculate expected measurement from particle to landmark
            float dx = landmark_x - particle_x;
            float dy = landmark_y - particle_y;
            float expected_range = sqrtf(dx*dx + dy*dy);
            float expected_bearing = atan2f(dy, dx) - particle_theta;

            // Normalize bearing to [-pi, pi]
            while (expected_bearing > M_PI) expected_bearing -= 2.0f * M_PI;
            while (expected_bearing < -M_PI) expected_bearing += 2.0f * M_PI;

            // Calculate likelihood using multivariate Gaussian
            float range_diff = meas.range - expected_range;
            float bearing_diff = meas.bearing - expected_bearing;

            // Normalize bearing difference to [-pi, pi]
            while (bearing_diff > M_PI) bearing_diff -= 2.0f * M_PI;
            while (bearing_diff < -M_PI) bearing_diff += 2.0f * M_PI;

            // Apply measurement model (likelihood)
            float range_prob = expf(-0.5f * powf(range_diff / range_std, 2)) /
                              (range_std * sqrtf(2.0f * M_PI));
            float bearing_prob = expf(-0.5f * powf(bearing_diff / bearing_std, 2)) /
                                (bearing_std * sqrtf(2.0f * M_PI));

            // Multiply probabilities (assume independence)
            weight *= range_prob * bearing_prob;
        }

        // Update particle weight
        weights[idx] = weight;
    }
}