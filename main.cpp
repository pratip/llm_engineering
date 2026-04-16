#include <iostream>
#include <vector>
#include <climits>
#include <chrono>

using namespace std;

// Constants for the LCG algorithm (Python default values)
const uint64_t LCG_A = 1664525;
const uint64_t LCG_C = 1013904223;
const uint64_t LCG_M = 0x100000000ULL; // 2^32

class LCG {
public:
    explicit LCG(uint64_t seed) : current_value(seed) {}

    // Generates next pseudo-random number in sequence
    uint64_t next() noexcept {
        current_value = (LCG_A * current_value + LCG_C) % LCG_M;
        return current_value;
    }

private:
    uint64_t current_value;
};

// Generates array of random numbers in specified range using LCG
vector<int> generate_random_numbers(int n, uint64_t seed, int min_val, int max_val) {
    LCG generator(seed);
    vector<int> numbers;
    numbers.reserve(n);

    int range = max_val - min_val + 1;

    for (int i = 0; i < n; ++i) {
        uint64_t raw_val = generator.next() % range;
        numbers.push_back(static_cast<int>(raw_val) + min_val);
    }

    return numbers;
}

// Finds maximum subarray sum using brute-force approach (O(n^2))
long long max_subarray_sum(const vector<int>& numbers) {
    int n = numbers.size();
    long long max_sum = LLONG_MIN;

    for (int i = 0; i < n; ++i) {
        long long current_sum = 0;
        for (int j = i; j < n; ++j) {
            current_sum += numbers[j];
            max_sum = max(max_sum, current_sum);
        }
    }

    return max_sum;
}

int main() {
    constexpr int NUM_ELEMENTS = 10000;
    constexpr uint64_t INITIAL_SEED = 42;
    constexpr int RANGE_MIN = -10;
    constexpr int RANGE_MAX = 10;

    auto start_time = chrono::high_resolution_clock::now();

    long long total_max_sum = 0;
    LCG seed_generator(INITIAL_SEED);

    for (int outer_loop = 0; outer_loop < 20; ++outer_loop) {
        uint64_t new_seed = seed_generator.next();
        vector<int> numbers = generate_random_numbers(NUM_ELEMENTS, new_seed, RANGE_MIN, RANGE_MAX);
        total_max_sum += max_subarray_sum(numbers);
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Total Maximum Subarray Sum (20 runs): " << total_max_sum << "\n";
    cout << "Execution Time: " << (duration.count() / 1000.0) << " seconds\n";

    return 0;
}