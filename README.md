# Particle Filter Localization with GPU Acceleration

A CUDA-accelerated implementation of a particle filter for robot localization, complete with performance analysis tools and real-time visualization.

## Overview

This project implements a particle filter algorithm for robot localization using CUDA for parallel processing on GPUs. It optimizes key data structures and algorithms for GPU computation to achieve significant speedups over traditional CPU implementations.

## Features

- **GPU-Accelerated Particle Filter**  
  Core algorithms implemented in CUDA for prediction, weight update, and resampling.  
- **Real-time Visualization**  
  Interactive display of particles, landmarks, and estimated pose using OpenCV.  
- **Performance Benchmarking**  
  Tools to measure execution times across different particle counts.  
- **Custom Data Structures**  
  - Flat, circular‐buffer linked list for ancestry tracking  
  - Circular queue for measurement history  
- **Performance Analysis**  
  Generate plots to visualize scaling behavior and component-wise breakdown.

## Dependencies

- CUDA Toolkit (11.0 or later)  
- OpenCV (4.2 or later)  
- C++17–compatible compiler  
- CMake (3.10 or later)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Visualization Mode

Run the particle filter with real-time visualization:

```bash
./particle_filter_localization
```
### Benchmark Mode
Run performance benchmarks and generate analysis plots:

```bash
./particle_filter_localization --benchmark
```
