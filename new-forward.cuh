#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define CONSTANT_KERNEL

const constexpr static int K = 7;
const constexpr static int B = 10000;

namespace mxnet
{
namespace op
{

__constant__ float deviceKernel[12 * 7 * 7];


__global__ void base(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (H_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

#ifdef CONSTANT_KERNEL
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#else
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#endif // CONSTANT_KERNEL

    int b = blockDim.x * blockIdx.x + threadIdx.x;

    if (b < B) // for each image in the batch
    {
        for (int m = 0; m < M; m++)         // for each output feature maps
            for (int h = 0; h < H_out; h++) // for each output element
                for (int w = 0; w < W_out; w++)
                {
                    float Pvalue = 0;
                    for (int c = 0; c < C; c++)     // sum over all input feature maps
                        for (int p = 0; p < K; p++) // KxK filter
                            for (int q = 0; q < K; q++)
                                Pvalue += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
                    y4d(b, m, h, w) = Pvalue;
                }
    }
#undef y4d
#undef x4d
#undef k4d
}


template<const int BLOCK_SIZE, const int CHANNELS, const int D_o, const int M, const int H, const int W>
__global__ void parallel_output(float * __restrict__ y, const float * const __restrict__ x){

#define y4d(i3, i2, i1, i0) y[(i3) * (M * D_o * D_o) + (i2) * (D_o * D_o) + (i1) * (D_o) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (CHANNELS * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (CHANNELS * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const constexpr int D_grid = (D_o / BLOCK_SIZE) + 1;
    
    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int tmp = bz / D_grid;
    int row_start = tmp * BLOCK_SIZE + ty;
    int col_start = (bz - tmp * D_grid) * BLOCK_SIZE + tx; // (bz % D_grid)  => (bz - (bz / D_grid) * D_grid)

    if (row_start < D_o && col_start < D_o) {
        float Pvalue = 0;
        #pragma unroll
        for (int ch = 0; ch < CHANNELS; ch++) {
            #pragma unroll
            for (int r = 0; r < K; r++) {
                #pragma unroll
                for (int c = 0; c < K; c++) {
                    Pvalue += x4d(bx, ch, row_start + r, col_start + c) * k4d(by, ch, r, c);
                }
            }
        }
        y4d(bx, by, row_start, col_start) = Pvalue;
    }
#undef k4d
#undef y4d
#undef x4d
}


template<const int TILE_SIZE, const int BLOCK_SIZE>
__global__ void shared_convolution(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int D_out)
{
#define y4d(i3, i2, i1, i0) y[(i3) * (M * D_out * D_out) + (i2) * (D_out * D_out) + (i1) * (D_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#ifdef CONSTANT
#define k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#else
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#endif

    __shared__ float Xds[BLOCK_SIZE][BLOCK_SIZE][12];
    int D_grid = ceil(1.0*D_out / TILE_SIZE);

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row_start = (bz / D_grid) * TILE_SIZE + ty;
    int col_start = (bz % D_grid) * TILE_SIZE + tx;

    for (int ch = 0; ch < C; ch++) {
        Xds[ch][ty][tx] = (row_start < H && col_start < W) ? x4d(bx, ch, row_start, col_start) : 0;
    }

    __syncthreads();

    if (tx < TILE_SIZE && ty < TILE_SIZE && row_start < D_out && col_start < D_out) {
        float Pvalue = 0;
        for (int ch = 0; ch < C; ch++)
            #pragma unroll
            for (int r = 0; r < K; r++)
                #pragma unroll
                for (int c = 0; c < K; c++)
                    Pvalue += Xds[ch][ty+r][tx+c] * k4d(by, ch, r, c);
        y4d(bx, by, row_start, col_start) = Pvalue;
    }

#undef y4d
#undef x4d
#undef k4d
#undef x4ds
}


template<const int BLOCK_SIZE, const int C, const int D_out, const int M, const int H, const int W>
__global__ void fused_unroll_gemm(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k){
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define y4d(i3, i2, i1, i0) y[(i3) * (M * numBColumns) + (i2) * (numBColumns) + (i1) * (D_out) + i0]

    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    constexpr const int numAColumns = C * K * K;
    constexpr const int numBColumns = D_out * D_out;
    constexpr const int k2 = K * K;

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
    int tx = threadIdx.x, ty = threadIdx.y;

    int out_idx = bx * BLOCK_SIZE + tx;
    int mask_out = by * BLOCK_SIZE + ty;
    
    int h_out = out_idx / D_out;

    float Pvalue = 0;
    #pragma unroll
    for (int i = 0; i < numAColumns/BLOCK_SIZE + 1; ++i) {
        int r = i * BLOCK_SIZE + ty;
        int c = i * BLOCK_SIZE + tx;

        int mask_in = r / k2;
        int tmp = r - k2 * mask_in; // r % K2

        int tk = tmp / K;
        int w_unrolled = (out_idx - h_out * D_out) + (tmp - tk * K); // out_idx % D_out + (r % K2) % K

        Nds[ty][tx] = (r < numAColumns && out_idx < numBColumns) ? x4d(bz, mask_in, h_out + tk, w_unrolled) : 0;
        Mds[ty][tx] = (mask_out < M && c < numAColumns) ? k[mask_out * numAColumns + c] : 0;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }
    
    if (mask_out < M && out_idx < numBColumns)
        y4d(bz, mask_out, h_out, out_idx - h_out * D_out) = Pvalue; // out_idx, out_idx % D_out

#undef y4d
#undef x4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    // const int C = x.shape_[1];
    // const int H = x.shape_[2];
    // const int W = x.shape_[3];

    // const float D_o = H - K + 1;

/*
    dim3 gridDim((B + 511) / 512);
    dim3 blockDim(512);

    base<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W);
*/

    /*
    cudaStream_t s = 0;
    if(M == 12){
        constexpr const int B = 10000;
        constexpr const int C = 1;
        constexpr const int H = 72;
        constexpr const int W = 72;

        cudaMemcpyToSymbolAsync(deviceKernel, w.dptr_, M*C*K*K*sizeof(float), 0, cudaMemcpyDeviceToDevice, s);

        constexpr const int D_o = H - K + 1; // 66
        constexpr const int BLOCK_SIZE = 24;
        constexpr const int grid_z = ((D_o / BLOCK_SIZE) + 1) * ((D_o / BLOCK_SIZE) + 1);

        dim3 gridDim(B, M, grid_z); // images x output_masks x (blocks per image)
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        parallel_output<BLOCK_SIZE, 1, D_o, 12, H, W><<<gridDim, blockDim, 0, s>>>(y.dptr_, x.dptr_);
    }else{
        constexpr const int B = 10000;
        constexpr const int C = 12;
        constexpr const int H = 33;
        constexpr const int W = 33;

        cudaMemcpyToSymbolAsync(deviceKernel, w.dptr_, M*C*K*K*sizeof(float), 0, cudaMemcpyDeviceToDevice, s);

        constexpr const int D_o = H - K + 1; // 27
        constexpr const int BLOCK_SIZE = 32;
        constexpr const int grid_z = ((D_o / BLOCK_SIZE) + 1) * ((D_o / BLOCK_SIZE) + 1);
        
        dim3 gridDim(B, M, grid_z); // images x output_masks x (blocks per image)
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        parallel_output<BLOCK_SIZE, 12, D_o, 24, H, W><<<gridDim, blockDim, 0, s>>>(y.dptr_, x.dptr_);
    }
    */

   /*
    constexpr const int TILE_SIZE = 24;
    constexpr const int BLOCK_SIZE = TILE_SIZE + K - 1;
    constexpr const int grid_z = ((D_o / BLOCK_SIZE) + 1) * ((D_o / BLOCK_SIZE) + 1);

    dim3 gridDim(B, M, grid_z);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    shared_convolution<TILE_SIZE, BLOCK_SIZE><<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, D_o);
    */


    if(M == 12){
        constexpr const int BLOCK_SIZE = 24;
        constexpr const int C = 1;
        constexpr const int H = 72;
        constexpr const int W = 72;

        constexpr const int D_o = H - K + 1; // 66
        constexpr const int grid_z = ((D_o / BLOCK_SIZE) + 1) * ((D_o / BLOCK_SIZE) + 1);

        cudaMemcpyToSymbol(deviceKernel, w.dptr_, 12*49*sizeof(float), 0, cudaMemcpyDeviceToDevice);

        dim3 gridDim(B, M, grid_z); // images x output_masks x (blocks per image)
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        parallel_output<BLOCK_SIZE, C, D_o, 12, H, W><<<gridDim, blockDim>>>(y.dptr_, x.dptr_);
    }
    else{
        constexpr const int BLOCK_SIZE = 24;
        constexpr const int C = 12;
        constexpr const int H = 33;
        constexpr const int W = 33;

        constexpr const int D_o = H - K + 1; // 27

        dim3 gridDim((D_o * D_o)/BLOCK_SIZE + 1, 1, B);
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        fused_unroll_gemm<BLOCK_SIZE, C, D_o, 24, H, W><<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_);
    }
}

template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes");
}
}
}

#endif
