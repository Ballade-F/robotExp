#ifndef RING_BUFFER_HPP
#define RING_BUFFER_HPP

#include <vector>
#include <stdexcept>

template <typename T>
class RingBuffer {
public:
    RingBuffer(size_t max_length) : max_length(max_length), next(0) {
        queue.reserve(max_length);
    }

    void push(const T& item) {
        if (queue.size() < max_length) {
            queue.push_back(item);
            next = (next + 1) % max_length;
        } else {
            queue[next] = item;
            next = (next + 1) % max_length;
        }
    }

    T operator[](int index) const {
        if (queue.empty()) {
            throw std::out_of_range("RingBuffer is empty");
        }
        if (index >= static_cast<int>(queue.size()) || index < -static_cast<int>(queue.size())) {
            throw std::out_of_range("index out of RingBuffer range");
        }
        int adjusted_index = (next - index - 1 + queue.size()) % queue.size();
        return queue[adjusted_index];
    }

    size_t size() const {
        return queue.size();
    }

private:
    size_t max_length;
    std::vector<T> queue;
    size_t next;
};

#endif // RING_BUFFER_HPP