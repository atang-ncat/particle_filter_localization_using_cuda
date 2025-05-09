#include "particle_filter/particle_filter.cuh"
#include "particle_filter/motion_model.cuh"
#include "particle_filter/measurement_model.cuh"
#include "data_structures/linked_list.cuh"
#include "data_structures/circular_queue.cuh"
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <iostream>

// Helper function for CUDA error checking
inline void cudaCheckError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) cudaCheckError(err, __FILE__, __LINE__)


__global__ void initParticlesKernel(
    float* x,
    float* y,
    float* theta,
    float* weights,
    float init_x,
    float init_y,
    float init_theta,
    float std_x,
    float std_y,
    float std_theta,
    curandState* rand_states,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        // Initialize random state
        curand_init(clock64(), idx, 0, &rand_states[idx]);

        // Generate random particle pose from Gaussian distribution
        float x_noise = curand_normal(&rand_states[idx]) * std_x;
        float y_noise = curand_normal(&rand_states[idx]) * std_y;
        float theta_noise = curand_normal(&rand_states[idx]) * std_theta;

        // Set initial particle state
        x[idx] = init_x + x_noise;
        y[idx] = init_y + y_noise;
        theta[idx] = init_theta + theta_noise;

        // Set uniform weight
        weights[idx] = 1.0f / num_particles;
    }
}

// Constructor
ParticleFilter::ParticleFilter(int num_particles) : num_particles(num_particles) {
    // Initialize particle arrays
    d_x.resize(num_particles);
    d_y.resize(num_particles);
    d_theta.resize(num_particles);
    d_weights.resize(num_particles);
    d_rand_states.resize(num_particles);

    // Initialize CUDA random states
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;
    unsigned long seed = time(NULL);

    // Check if d_rand_states is properly sized
    if (d_rand_states.size() != num_particles) {
        std::cerr << "Error: d_rand_states size mismatch" << std::endl;
        return;
    }

    // Initialize random states
    initRandStatesKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_rand_states.data()),
        seed,
        num_particles
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error initializing random states: " << cudaGetErrorString(err) << std::endl;
    }

    // Initialize weights to uniform
    thrust::fill(thrust::device, d_weights.begin(), d_weights.end(), 1.0f / num_particles);

    // Initialize particle_ancestry and measurement_history to nullptr first
    particle_ancestry = nullptr;
    measurement_history = nullptr;

    try {
        // Create and initialize particle ancestry
        particle_ancestry = new LinkedList(num_particles);
        // Make sure the linked list gets properly initialized
        if (particle_ancestry != nullptr) {
            particle_ancestry->initialize();
        }

        // Create measurement history
        measurement_history = new CircularQueue(50);
    } catch (const std::exception& e) {
        std::cerr << "Error initializing data structures: " << e.what() << std::endl;
        // Clean up
        delete particle_ancestry;
        particle_ancestry = nullptr;
        delete measurement_history;
        measurement_history = nullptr;
        throw;
    }

    // Default parameters
    wheelbase = 2.0f;
    range_std_ = 0.5f;
    bearing_std_ = 0.02f;
    update_count_ = 0;
}

// Destructor
ParticleFilter::~ParticleFilter() {
    if (particle_ancestry != nullptr) {
        delete particle_ancestry;
        particle_ancestry = nullptr;
    }

    if (measurement_history != nullptr) {
        delete measurement_history;
        measurement_history = nullptr;
    }
}

// Initialize particles around a given pose
void ParticleFilter::initialize(float x, float y, float theta,
                              float std_x, float std_y, float std_theta) {
    // Call initialization kernel
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    initParticlesKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_theta.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        x, y, theta, std_x, std_y, std_theta,
        thrust::raw_pointer_cast(d_rand_states.data()),
        num_particles
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Re-initialize ancestry tracking if needed
    if (particle_ancestry == nullptr) {
        try {
            particle_ancestry = new LinkedList(num_particles);
        } catch (const std::exception& e) {
            std::cerr << "Failed to create LinkedList: " << e.what() << std::endl;
            return;
        }
    }

    // Make sure to call initialize
    if (particle_ancestry != nullptr) {
        particle_ancestry->initialize();
    }
}


void ParticleFilter::updateWeights(
    const std::vector<Measurement>& measurements,
    const std::vector<float>& landmarks_x,
    const std::vector<float>& landmarks_y
) {
    if (measurements.empty()) return;

    // Copy measurements to device
    thrust::device_vector<Measurement> d_measurements = measurements;

    // Copy landmark positions to device if not already cached
    if (d_landmarks_x.size() != landmarks_x.size()) {
        d_landmarks_x = landmarks_x;
        d_landmarks_y = landmarks_y;
    }

    // Store measurements in history queue for future use
    for (const auto& m : measurements) {
        measurement_history->push(m);
    }

    // Launch weight update kernel
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    updateWeightsKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_theta.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_landmarks_x.data()),
        thrust::raw_pointer_cast(d_landmarks_y.data()),
        thrust::raw_pointer_cast(d_measurements.data()),
        static_cast<int>(measurements.size()),
        range_std_,
        bearing_std_,
        num_particles,
        static_cast<int>(landmarks_x.size())
    );

    // Normalize weights to prevent numerical underflow
    float sum = thrust::reduce(thrust::device, d_weights.begin(), d_weights.end(), 0.0f);
    if (sum > 0) {
        thrust::transform(thrust::device, d_weights.begin(), d_weights.end(), d_weights.begin(),
                         [sum] __device__ (float w) { return w / sum; });
    } else {
        // All weights are zero, reset to uniform
        thrust::fill(thrust::device, d_weights.begin(), d_weights.end(), 1.0f / num_particles);
    }
}

void ParticleFilter::getBestEstimate(float& x, float& y, float& theta) const {
    // Use host vectors to avoid const issues
    thrust::host_vector<float> h_weights = d_weights;

    // Find max element index on host
    auto max_iter = std::max_element(h_weights.begin(), h_weights.end());
    int best_idx = max_iter - h_weights.begin();

    // Copy values
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_y = d_y;
    thrust::host_vector<float> h_theta = d_theta;

    x = h_x[best_idx];
    y = h_y[best_idx];
    theta = h_theta[best_idx];
}

// In particle_filter.cu
void ParticleFilter::visualize(cv::Mat& image, float scale) const {
    // Copy particle data to host
    thrust::host_vector<float> h_x = d_x;
    thrust::host_vector<float> h_y = d_y;
    thrust::host_vector<float> h_theta = d_theta;
    thrust::host_vector<float> h_weights = d_weights;

    // Find max weight for normalization
    float max_weight = *thrust::max_element(h_weights.begin(), h_weights.end());

    // Draw particles
    for (int i = 0; i < num_particles; i++) {
        // Scale coordinates to image space
        int x = static_cast<int>(h_x[i] * scale);
        int y = static_cast<int>(h_y[i] * scale);

        // Check if within image bounds
        if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
            // Normalize weight for color intensity (0-255)
            int intensity = static_cast<int>(255 * h_weights[i] / max_weight);

            // Draw particle as a colored point
            cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, intensity, 255-intensity), -1);

            // Draw orientation line
            int endpoint_x = static_cast<int>(x + 10 * cosf(h_theta[i]));
            int endpoint_y = static_cast<int>(y + 10 * sinf(h_theta[i]));
            cv::line(image, cv::Point(x, y), cv::Point(endpoint_x, endpoint_y),
                    cv::Scalar(0, 255, 0), 1);
        }
    }

    // Draw best estimate
    float best_x, best_y, best_theta;
    getBestEstimate(best_x, best_y, best_theta);

    int best_x_px = static_cast<int>(best_x * scale);
    int best_y_px = static_cast<int>(best_y * scale);

    if (best_x_px >= 0 && best_x_px < image.cols && best_y_px >= 0 && best_y_px < image.rows) {
        // Draw best estimate position
        cv::circle(image, cv::Point(best_x_px, best_y_px), 5, cv::Scalar(0, 0, 255), -1);

        // Draw orientation line
        int endpoint_x = static_cast<int>(best_x_px + 20 * cosf(best_theta));
        int endpoint_y = static_cast<int>(best_y_px + 20 * sinf(best_theta));
        cv::line(image, cv::Point(best_x_px, best_y_px),
                cv::Point(endpoint_x, endpoint_y), cv::Scalar(0, 0, 255), 2);
    }
}

void ParticleFilter::setLandmarks(const std::vector<float>& landmarks_x,
                                const std::vector<float>& landmarks_y) {
    d_landmarks_x = landmarks_x;
    d_landmarks_y = landmarks_y;
}

// Update the particle filter with new measurements and motion
void ParticleFilter::update(
    float delta_t,
    float velocity,
    float steering_angle,
    const std::vector<Measurement>& measurements,
    const std::vector<float>& landmarks_x,
    const std::vector<float>& landmarks_y
) {
    // 1. Prediction step - move particles according to motion model
    predict(delta_t, velocity, steering_angle);

    // 2. Update weights based on measurements
    updateWeights(measurements, landmarks_x, landmarks_y);

    // 3. Resample particles based on weights
    resample();

    // Increment update counter
    update_count_++;
}

// Calculate effective sample size (ESS) to monitor filter performance
float ParticleFilter::getEffectiveSampleSize() const {
    // Make a non-const copy to work with
    thrust::device_vector<float> weights_copy = d_weights;

    // ESS = 1 / sum(w_i^2) where w_i are normalized weights
    thrust::device_vector<float> squared_weights(num_particles);
    thrust::transform(weights_copy.begin(), weights_copy.end(), squared_weights.begin(),
                     [] __device__ (float w) { return w * w; });

    float sum_squared = thrust::reduce(thrust::device, squared_weights.begin(),
                                     squared_weights.end(), 0.0f);

    return (sum_squared > 0.0f) ? 1.0f / sum_squared : 0.0f;
}

// Get all particle states for visualization or analysis
void ParticleFilter::getParticleStates(std::vector<float>& x, std::vector<float>& y,
                                     std::vector<float>& theta) const {
    x.resize(num_particles);
    y.resize(num_particles);
    theta.resize(num_particles);

    // Copy from device to host
    thrust::copy(d_x.begin(), d_x.end(), x.begin());
    thrust::copy(d_y.begin(), d_y.end(), y.begin());
    thrust::copy(d_theta.begin(), d_theta.end(), theta.begin());
}

// Get weight statistics for debugging and monitoring
// Get weight statistics for debugging and monitoring
void ParticleFilter::getWeightStats(float& min_weight, float& max_weight, float& avg_weight) const {
    // Copy to host to avoid const issues
    thrust::host_vector<float> h_weights = d_weights;

    // Find min and max on host instead of device
    auto min_iter = std::min_element(h_weights.begin(), h_weights.end());
    auto max_iter = std::max_element(h_weights.begin(), h_weights.end());

    min_weight = *min_iter;
    max_weight = *max_iter;

    // Calculate average
    float sum = 0.0f;
    for (const auto& w : h_weights) {
        sum += w;
    }
    avg_weight = sum / num_particles;
}


// Get ancestry path for a particle
std::vector<int> ParticleFilter::getAncestryPath(int particle_idx, int path_length) const {
    std::vector<int> ancestry(path_length);
    particle_ancestry->getAncestry(particle_idx, ancestry.data(), path_length);
    return ancestry;
}