#include "particle_filter/resampling.cuh"
#include "particle_filter/particle_filter.cuh"
#include "data_structures/linked_list.cuh"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <curand_kernel.h>
#include <iostream>

// Helper function for CUDA error checking
inline void cudaCheckError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) cudaCheckError(err, __FILE__, __LINE__)

__global__ void systematicResamplingKernel(
    float* x_in,
    float* y_in,
    float* theta_in,
    float* weights,
    float* x_out,
    float* y_out,
    float* theta_out,
    int* ancestry_index,
    float random_start,
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        float u = (random_start + idx) / num_particles;

        // Binary search to find the particle to sample
        int j = 0;
        float cumulative_sum = weights[0];

        while (u > cumulative_sum && j < num_particles - 1) {
            j++;
            cumulative_sum += weights[j];
        }

        // Copy the selected particle's state
        x_out[idx] = x_in[j];
        y_out[idx] = y_in[j];
        theta_out[idx] = theta_in[j];

        // Store ancestry index
        ancestry_index[idx] = j;
    }
}

void ParticleFilter::resample() {
    std::cout << "Starting resampling..." << std::endl;
    // Check if particle_ancestry is initialized
    if (particle_ancestry == nullptr) {
        std::cerr << "Error: particle_ancestry is null in resample()" << std::endl;
        try {
            std::cout << "Attempting to recreate particle_ancestry..." << std::endl;
            particle_ancestry = new LinkedList(num_particles);
            particle_ancestry->initialize();
        } catch (const std::exception& e) {
            std::cerr << "Failed to create LinkedList: " << e.what() << std::endl;
            return;
        }
    }

    // Create temporary vectors for resampled particles
    thrust::device_vector<float> d_x_resampled(num_particles);
    thrust::device_vector<float> d_y_resampled(num_particles);
    thrust::device_vector<float> d_theta_resampled(num_particles);

    // Create array for ancestry indices
    thrust::device_vector<int> d_ancestry_indices(num_particles, 0);

    // Host vector for safely transferring ancestry data
    thrust::host_vector<int> h_ancestry_indices(num_particles, 0);

    // Generate a random starting point for systematic resampling
    float random_start = static_cast<float>(rand()) / RAND_MAX;

    // Launch the resampling kernel
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    systematicResamplingKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_theta.data()),
        thrust::raw_pointer_cast(d_weights.data()),
        thrust::raw_pointer_cast(d_x_resampled.data()),
        thrust::raw_pointer_cast(d_y_resampled.data()),
        thrust::raw_pointer_cast(d_theta_resampled.data()),
        thrust::raw_pointer_cast(d_ancestry_indices.data()),
        random_start,
        num_particles
    );

    // Make sure kernel execution is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy ancestry indices to host memory for safer handling
    thrust::copy(d_ancestry_indices.begin(), d_ancestry_indices.end(), h_ancestry_indices.begin());

    try {
        // Update ancestry tracking with host memory pointer
        particle_ancestry->update(h_ancestry_indices.data());
    } catch (const std::exception& e) {
        std::cerr << "Exception in particle_ancestry->update: " << e.what() << std::endl;
    }

    // Replace original particles with resampled ones
    d_x = d_x_resampled;
    d_y = d_y_resampled;
    d_theta = d_theta_resampled;

    // Reset weights to uniform
    thrust::fill(thrust::device, d_weights.begin(), d_weights.end(), 1.0f / num_particles);
}