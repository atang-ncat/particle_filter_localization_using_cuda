#include "data_structures/circular_queue.cuh"

CircularQueue::CircularQueue(int capacity)
	: capacity(capacity), size(0), front(0), rear(0) {
	measurements = new Measurement[capacity];
}

CircularQueue::~CircularQueue() {
	delete[] measurements;
}

void CircularQueue::push(const Measurement& measurement) {
	measurements[rear] = measurement;
	rear = (rear + 1) % capacity;

	if (size < capacity) {
		size++;
	} else {
		// Queue is full, move front pointer
		front = (front + 1) % capacity;
	}
}

Measurement* CircularQueue::getData() const {
	return measurements;
}