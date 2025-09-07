#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

// CUDA カーネル
__global__ void temporal_denoise_kernel(uint8_t** frames, int numFrames,
    int center, uint8_t* out,
    int w, int h, int stride,
    float alphaLow, float alphaMid, float alphaHigh,
    float strength)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * stride + x;

    // 時間平均
    float sum = 0.0f;
    for (int i = 0; i < numFrames; i++) {
        sum += frames[i][idx];
    }
    float avg = sum / numFrames;
    float cur = frames[center][idx];

    // 周波数別の重み付け
    float low = (1.0f - alphaLow) * cur + alphaLow * avg;
    float mid = (1.0f - alphaMid) * cur + alphaMid * avg;
    float high = (1.0f - alphaHigh) * cur + alphaHigh * avg;

    float denoised = (low + mid + high) / 3.0f;

    // 原画とのブレンド
    float outVal = (1.0f - strength) * cur + strength * denoised;

    out[idx] = (uint8_t)fminf(fmaxf(outVal, 0.0f), 255.0f);
}

extern "C" void runTemporalDenoise(
    const std::vector<const uint8_t*>& srcFrames,
    const std::vector<int>& strides,
    uint8_t* dst, int w, int h, int dstStride,
    int radius,
    float alphaLow, float alphaMid, float alphaHigh,
    float strength)
{
    int numFrames = (int)srcFrames.size();
    int center = radius;

    size_t size = h * dstStride;

    // GPU メモリ確保
    std::vector<uint8_t*> d_frames(numFrames);
    for (int i = 0; i < numFrames; i++) {
        cudaMalloc(&d_frames[i], size);
        cudaMemcpy(d_frames[i], srcFrames[i], size, cudaMemcpyHostToDevice);
    }

    uint8_t* d_out;
    cudaMalloc(&d_out, size);

    // フレームポインタ配列を GPU に転送
    uint8_t** d_frame_ptrs;
    cudaMalloc(&d_frame_ptrs, numFrames * sizeof(uint8_t*));
    cudaMemcpy(d_frame_ptrs, d_frames.data(), numFrames * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
        (h + block.y - 1) / block.y);

    // カーネル呼び出し
    temporal_denoise_kernel << <grid, block >> > (d_frame_ptrs, numFrames, center,
        d_out, w, h, dstStride,
        alphaLow, alphaMid, alphaHigh,
        strength);

    cudaMemcpy(dst, d_out, size, cudaMemcpyDeviceToHost);

    // 後始末
    for (int i = 0; i < numFrames; i++) cudaFree(d_frames[i]);
    cudaFree(d_out);
    cudaFree(d_frame_ptrs);
}
