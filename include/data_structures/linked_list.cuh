#ifndef LINKED_LIST_CUH
#define LINKED_LIST_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

class LinkedList {
private:
    int num_particles;
    thrust::device_vector<int> d_ancestry;
    int history_length;
    int current_index;

public:
    LinkedList(int num_particles, int history_length = 10);
    ~LinkedList();

    void initialize();
    void update(int* ancestry_indices);
    void getAncestry(int particle_idx, int* ancestry_array, int path_length) const;
};

#endif // LINKED_LIST_CUH