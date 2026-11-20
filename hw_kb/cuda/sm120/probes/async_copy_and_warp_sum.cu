#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void stage_partial(const int* input, int* output) {
    __shared__ __align__(16) int stage[32][4];
    const unsigned lane = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        stage[lane][i] = -99;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(stage[lane]));
    const unsigned valid_bytes = (lane % 4) * sizeof(int);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(input + lane * 4), "r"(valid_bytes) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        output[lane * 4 + i] = stage[lane][i];
    }
}

__global__ void warp_sum(const int* input, float* output) {
    const unsigned lane = threadIdx.x;
    const unsigned i = blockIdx.x * 32 + lane;
    float value = static_cast<float>(input[i]);
    for (int delta = 1; delta < 32; delta *= 2) {
        value += __shfl_xor_sync(0xffffffff, value, delta);
    }
    output[i] = value;
}

int main() {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 0) {
        std::fprintf(stderr, "This probe certifies SM120 only\n");
        return 1;
    }

    std::array<int, 128> input{}, copied{};
    std::array<float, 128> reduced{};
    for (int i = 0; i < 128; ++i) {
        input[i] = (i * 7) % 29 - 14;
    }
    int* device_input;
    int* device_copied;
    float* device_reduced;
    check(cudaMalloc(&device_input, sizeof(input)));
    check(cudaMalloc(&device_copied, sizeof(copied)));
    check(cudaMalloc(&device_reduced, sizeof(reduced)));
    check(cudaMemcpy(device_input, input.data(), sizeof(input), cudaMemcpyHostToDevice));

    stage_partial<<<1, 32>>>(device_input, device_copied);
    check(cudaGetLastError());
    check(cudaMemcpy(copied.data(), device_copied, sizeof(copied), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 128; ++i) {
        const int expected = i % 4 < (i / 4) % 4 ? input[i] : 0;
        if (copied[i] != expected) {
            std::puts("PROBE FAIL max_diff=n/a # partial_copy");
            return 1;
        }
    }
    std::puts("PROBE PASS max_diff=0.0 # partial_copy");

    warp_sum<<<4, 32>>>(device_input, device_reduced);
    check(cudaGetLastError());
    check(cudaMemcpy(reduced.data(), device_reduced, sizeof(reduced), cudaMemcpyDeviceToHost));
    for (int block = 0; block < 4; ++block) {
        float expected = 0;
        for (int i = 0; i < 32; ++i) {
            expected += input[block * 32 + i];
        }
        for (int i = 0; i < 32; ++i) {
            if (reduced[block * 32 + i] != expected) {
                std::puts("PROBE FAIL max_diff=n/a # warp_sum");
                return 1;
            }
        }
    }
    std::puts("PROBE PASS max_diff=0.0 # warp_sum");
    check(cudaFree(device_input));
    check(cudaFree(device_copied));
    check(cudaFree(device_reduced));
}
