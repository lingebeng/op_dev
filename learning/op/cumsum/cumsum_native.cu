#include <cuda_runtime.h>
#include <iostream>
#include <vector>

constexpr int kWarpSize = 32;
constexpr int kMaxWarps = 32;  // 支持最大 1024 线程 (32 warps)

// Warp 内 inclusive prefix sum (Kogge-Stone 算法)
// 完全在寄存器中通过 shuffle 指令完成，零共享内存开销
__device__ __forceinline__ float warp_prefix_sum(float val, int lane_id) {
    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
        float neighbor = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) {
            val += neighbor;
        }
    }
    return val;
}

// Block 内 inclusive prefix sum
// 三阶段：warp 内 scan → warp 间 scan → 广播补偿
__global__ void block_cumsum_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid >= n) return;

    int lane_id = tid % kWarpSize;
    int warp_id = tid / kWarpSize;
    int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

    __shared__ float warp_sums[kMaxWarps];

    // 阶段一：Warp 内 prefix sum
    float val = warp_prefix_sum(input[gid], lane_id);

    // 阶段二：每个 Warp 的最后一个线程把该 Warp 总和写入共享内存
    if (lane_id == kWarpSize - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 阶段三：Warp 0 对所有 Warp 总和做 prefix sum
    if (warp_id == 0) {
        float warp_val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        warp_val = warp_prefix_sum(warp_val, lane_id);
        if (lane_id < num_warps) {
            warp_sums[lane_id] = warp_val;
        }
    }
    __syncthreads();

    // 阶段四：非首 Warp 的线程加上前面所有 Warp 的累积和
    if (warp_id > 0) {
        val += warp_sums[warp_id - 1];
    }

    output[gid] = val;
}

int main() {
    constexpr int N = 256;
    constexpr int kBlockSize = 256;
    constexpr size_t bytes = N * sizeof(float);

    // Host 端初始化：全 1 输入，期望输出 1, 2, 3, ..., 256
    std::vector<float> h_input(N, 1.0f);
    std::vector<float> h_output(N, 0.0f);

    // Device 端分配显存
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // 启动 Kernel
    int grid_size = (N + kBlockSize - 1) / kBlockSize;
    block_cumsum_kernel<<<grid_size, kBlockSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // 拷贝结果回 Host
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // 验证
    std::cout << "Output: ";
    for (int i = 0; i < 5; i++) std::cout << h_output[i] << " ";
    std::cout << "... " << h_output[N - 1] << std::endl;

    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (h_output[i] != static_cast<float>(i + 1)) {
            std::cerr << "Mismatch at [" << i << "]: "
                      << h_output[i] << " != " << i + 1 << std::endl;
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
