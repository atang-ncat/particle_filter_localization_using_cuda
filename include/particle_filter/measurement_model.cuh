#ifndef MEASUREMENT_MODEL_CUH
#define MEASUREMENT_MODEL_CUH

#include <cuda_runtime.h>
#include "particle_filter/particle_filter.cuh"  // For Measurement struct

/**
 * CUDA kernel for updating particle weights based on measurements
 */
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
);

#endif // MEASUREMENT_MODEL_CUH