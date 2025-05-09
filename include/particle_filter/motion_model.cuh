#ifndef MOTION_MODEL_CUH
#define MOTION_MODEL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void initRandStatesKernel(curandState* states, unsigned long seed, int num_particles);

/**
 * CUDA kernel implementing the bicycle motion model for particle prediction
 */
__global__ void predictParticlesKernel(
    float* x,
    float* y,
    float* theta,
    float delta_t,
    float velocity,
    float steering_angle,
    float wheelbase,
    curandState* rand_states,
    float std_pos_x, float std_pos_y, float std_pos_theta,
    int num_particles
);

#endif // MOTION_MODEL_CUH