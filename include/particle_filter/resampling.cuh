#ifndef RESAMPLING_CUH
#define RESAMPLING_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

/**
 * CUDA kernel implementing systematic resampling for particles
 */
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
);

#endif // RESAMPLING_CUH
