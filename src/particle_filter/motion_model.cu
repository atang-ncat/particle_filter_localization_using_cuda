#include "particle_filter/motion_model.cuh"
#include "particle_filter/particle_filter.cuh"
#include <math.h>

__global__ void initRandStatesKernel(curandState* states, unsigned long seed, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void predictParticlesKernel(
    float* x,
    float* y,
    float* theta,
    float delta_t,
    float velocity,
    float steering_angle,
    float wheelbase,
    curandState* rand_states,
    float std_pos_x, float std_pos_y, float std_pos_theta, // Individual parameters instead of array
    int num_particles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_particles) {
        // Get thread's random state
        curandState local_state = rand_states[idx];

        // Apply bicycle motion model
        float theta_t = theta[idx];
        float new_x, new_y, new_theta;

        if (fabs(steering_angle) < 0.001f) {
            // Moving straight
            new_x = x[idx] + velocity * delta_t * cosf(theta_t);
            new_y = y[idx] + velocity * delta_t * sinf(theta_t);
            new_theta = theta_t;
        } else {
            // Turning motion
            float turning_radius = wheelbase / tanf(steering_angle);
            float angular_velocity = velocity / turning_radius;
            float angle_change = angular_velocity * delta_t;

            new_x = x[idx] + turning_radius * (sinf(theta_t + angle_change) - sinf(theta_t));
            new_y = y[idx] + turning_radius * (cosf(theta_t) - cosf(theta_t + angle_change));
            new_theta = fmodf(theta_t + angle_change, 2.0f * M_PI);

            // Normalize angle to [-pi, pi]
            if (new_theta > M_PI) new_theta -= 2.0f * M_PI;
            if (new_theta < -M_PI) new_theta += 2.0f * M_PI;
        }

        // Add noise using individual std parameters
        float x_noise = curand_normal(&local_state) * std_pos_x;
        float y_noise = curand_normal(&local_state) * std_pos_y;
        float theta_noise = curand_normal(&local_state) * std_pos_theta;

        x[idx] = new_x + x_noise;
        y[idx] = new_y + y_noise;
        theta[idx] = new_theta + theta_noise;

        // Save random state
        rand_states[idx] = local_state;
    }
}

// Implementation for ParticleFilter::predict method
void ParticleFilter::predict(float delta_t, float velocity, float steering_angle) {
    if (d_rand_states.size() != num_particles) {
        std::cerr << "Error: d_rand_states size mismatch with num_particles" << std::endl;
        return;
    }

    // Noise parameters as individual values rather than an array
    float std_pos_x = 0.1f;
    float std_pos_y = 0.1f;
    float std_pos_theta = 0.05f;

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    predictParticlesKernel<<<numBlocks, blockSize>>>(
        thrust::raw_pointer_cast(d_x.data()),
        thrust::raw_pointer_cast(d_y.data()),
        thrust::raw_pointer_cast(d_theta.data()),
        delta_t,
        velocity,
        steering_angle,
        wheelbase,
        thrust::raw_pointer_cast(d_rand_states.data()),
        std_pos_x, std_pos_y, std_pos_theta, // Pass individual values
        num_particles
    );

    // Check for errors
    cudaDeviceSynchronize();

    // Add error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in predict: " << cudaGetErrorString(err) << std::endl;
    }
}