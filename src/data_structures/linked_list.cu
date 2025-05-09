#include "data_structures/linked_list.cuh"
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <algorithm>

LinkedList::LinkedList(int num_particles, int history_length)
 : num_particles(num_particles), history_length(history_length), current_index(0) {
 std::cout << "Creating LinkedList with " << num_particles << " particles and "
           << history_length << " history length ("
           << (num_particles * history_length * sizeof(int)) << " bytes)" << std::endl;
 d_ancestry.resize(num_particles * history_length, -1);
}

LinkedList::~LinkedList() {
}

void LinkedList::initialize() {
 // Initialize all ancestry indices to -1 (no ancestors)
 thrust::fill(d_ancestry.begin(), d_ancestry.end(), -1);
 current_index = 0;
}

void LinkedList::getAncestry(int particle_idx, int* ancestry_path, int steps) const {
 if (particle_idx < 0 || particle_idx >= num_particles) {
  std::cerr << "Error: particle_idx out of range: " << particle_idx << std::endl;
  std::fill(ancestry_path, ancestry_path + steps, -1);
  return;
 }

 // Create a host copy of the device ancestry data
 thrust::host_vector<int> h_ancestry = d_ancestry;

 int idx = particle_idx;

 // For each step back in history
 for (int i = 0; i < steps; i++) {
  // Current history index minus i, wrapped around if needed
  int hist_idx = (current_index - i + history_length) % history_length;
  int array_idx = hist_idx * num_particles + idx;

  if (array_idx < 0 || array_idx >= h_ancestry.size()) {
   std::cerr << "Error: array_idx out of range: " << array_idx << std::endl;
   std::fill(ancestry_path + i, ancestry_path + steps, -1);
   break;
  }

  // Get the ancestor index from the host vector
  ancestry_path[i] = h_ancestry[array_idx];

  // Update idx to be the ancestor for the next iteration
  idx = ancestry_path[i];

  // If we hit -1, we've reached the end of the ancestry
  if (idx == -1 || idx < 0 || idx >= num_particles) {
   // Fill the rest with -1
   std::fill(ancestry_path + i + 1, ancestry_path + steps, -1);
   break;
  }
 }
}

void LinkedList::update(int* ancestry_indices) {
    // Make sure ancestry_indices isn't null
    if (ancestry_indices == nullptr) {
     std::cerr << "Error: ancestry_indices is null" << std::endl;
     return;
    }
   
    std::cout << "LinkedList::update - num_particles: " << num_particles
              << ", history_length: " << history_length
              << ", d_ancestry size: " << d_ancestry.size()
              << ", current_index: " << current_index << std::endl;
    
    size_t target_offset = current_index * num_particles;
    size_t write_end = target_offset + num_particles;
    
    if (write_end > d_ancestry.size()) {
     std::cerr << "Error: Attempting to write past the end of d_ancestry. "
               << "Offset: " << target_offset
               << ", Required: " << num_particles
               << ", Available: " << (d_ancestry.size() - target_offset) << std::endl;
     return;
    }
    
    std::cout << "First few ancestry indices: ";
    for (int i = 0; i < std::min(5, num_particles); ++i) {
     std::cout << ancestry_indices[i] << " ";
    }
    std::cout << std::endl;
    
    try {
     // Use a safer approach to copy the data
     thrust::host_vector<int> h_temp(num_particles);
     
     // Copy the data safely
     for (int i = 0; i < num_particles; i++) {
       h_temp[i] = ancestry_indices[i];
     }
     
     // Now copy to device vector
     thrust::copy(
         h_temp.begin(),
         h_temp.end(),
         d_ancestry.begin() + current_index * num_particles);
     
     current_index = (current_index + 1) % history_length;
     
     std::cout << "Update completed successfully. New current_index: " << current_index << std::endl;
    } catch (const std::exception& e) {
     std::cerr << "Error in LinkedList::update: " << e.what() << std::endl;
    }
   }