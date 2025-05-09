#ifndef CIRCULAR_QUEUE_CUH
#define CIRCULAR_QUEUE_CUH

#include "particle_filter/particle_filter.cuh"  // For Measurement struct

class CircularQueue {
private:
    int capacity;
    int size;
    int front;
    int rear;
    Measurement* measurements;

public:
    CircularQueue(int capacity);
    ~CircularQueue();

    void push(const Measurement& measurement);
    Measurement* getData() const;
    int getSize() const { return size; }
    int getCapacity() const { return capacity; }
};

#endif // CIRCULAR_QUEUE_CUH