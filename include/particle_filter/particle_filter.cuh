#ifndef PARTICLE_FILTER_CUH
#define PARTICLE_FILTER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <opencv2/opencv.hpp>
#include <vector>

// Forward declarations
class LinkedList;
class CircularQueue;

struct Measurement {
    float range;        // Range (distance) measurement
    float bearing;      // Bearing (angle) measurement
    int landmark_id;    // ID of the landmark
};

class ParticleFilter {
private:
    // Number of particles
    int num_particles;

    // Particle state on GPU (structure of arrays for coalesced access)
    thrust::device_vector<float> d_x;        // x positions
    thrust::device_vector<float> d_y;        // y positions
    thrust::device_vector<float> d_theta;    // orientations
    thrust::device_vector<float> d_weights;  // weights

    // Data structures
    LinkedList* particle_ancestry;           // Tracks particle history
    CircularQueue* measurement_history;      // Stores recent measurements

    // CUDA utilities
    thrust::device_vector<curandState> d_rand_states;

    // Map data
    thrust::device_vector<float> d_landmarks_x;
    thrust::device_vector<float> d_landmarks_y;

    // Motion model parameters
    float wheelbase;  // For bicycle model

    // Measurement model parameters
    float range_std_;    // Standard deviation for range measurements
    float bearing_std_;  // Standard deviation for bearing measurements

    // Update counter
    int update_count_;

public:
    // Constructor/destructor
    ParticleFilter(int num_particles);
    ~ParticleFilter();

    // Core particle filter operations
    void initialize(float x, float y, float theta,
                  float std_x, float std_y, float std_theta);
    void predict(float delta_t, float velocity, float steering_angle);
    void updateWeights(const std::vector<Measurement>& measurements,
                     const std::vector<float>& landmarks_x,
                     const std::vector<float>& landmarks_y);
    void resample();

    // Complete filter update (prediction, weight update, resampling)
    void update(float delta_t,
              float velocity,
              float steering_angle,
              const std::vector<Measurement>& measurements,
              const std::vector<float>& landmarks_x,
              const std::vector<float>& landmarks_y);

    // Data access methods
    void getBestEstimate(float& x, float& y, float& theta) const;
    float getEffectiveSampleSize() const;
    void getParticleStates(std::vector<float>& x, std::vector<float>& y, std::vector<float>& theta) const;
    void getWeightStats(float& min_weight, float& max_weight, float& avg_weight) const;
    std::vector<int> getAncestryPath(int particle_idx, int path_length) const;

    // Visualization
    void visualize(cv::Mat& image, float scale = 1.0f) const;

    // Helper methods
    void setLandmarks(const std::vector<float>& landmarks_x,
                    const std::vector<float>& landmarks_y);
};

#endif // PARTICLE_FILTER_CUH