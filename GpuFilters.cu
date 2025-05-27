#include <cuda_runtime.h>
#include <opencv2/core.hpp>

__global__ void dummyKernel() {}

void RunDummyCudaKernel() {
    dummyKernel << <1, 1 >> > ();
    cudaDeviceSynchronize();
}


__global__ void sobelKernel(const uchar* input, uchar* output, int width, int height, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;

    int gx = 0;
    int gy = 0;

    // grayscale from BGR
    auto getGray = [&](int xx, int yy) -> int {
        int offset = yy * step + xx * 3;
        return (int)(0.299f * input[offset + 2] + 0.587f * input[offset + 1] + 0.114f * input[offset]);
        };

    // Sobel X and Y kernels
    int sx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int sy[3][3] = { { 1,  2,  1}, {0, 0, 0}, {-1, -2, -1} };

    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            int gray = getGray(x + i, y + j);
            gx += gray * sx[j + 1][i + 1];
            gy += gray * sy[j + 1][i + 1];
        }

    int mag = min(255, abs(gx) + abs(gy));

    int offset = y * step + x * 3;
    output[offset + 0] = mag; // B
    output[offset + 1] = mag; // G
    output[offset + 2] = mag; // R
}

cv::Mat ApplyCudaFilter(const cv::Mat& input)
{
    cv::Mat output(input.size(), input.type());

    uchar* d_input = nullptr;
    uchar* d_output = nullptr;
    size_t size = input.step * input.rows;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((input.cols + 15) / 16, (input.rows + 15) / 16);
    sobelKernel<<<grid, block>>>(d_input, d_output, input.cols, input.rows, input.step);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}

__global__ void blurKernel(const uchar* input, uchar* output, int width, int height, int step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        for (int c = 0; c < 3; ++c)
        {
            int sum = 0;
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i)
                {
                    int idx = (y + j) * step + (x + i) * 3 + c;
                    sum += input[idx];
                }

            int outIdx = y * step + x * 3 + c;
            output[outIdx] = sum / 9;
        }
    }
}


cv::Mat ApplyCudaBlur(const cv::Mat& input)
{
    cv::Mat output(input.size(), input.type());

    uchar* d_input = nullptr;
    uchar* d_output = nullptr;
    size_t size = input.step * input.rows;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((input.cols + 15) / 16, (input.rows + 15) / 16);

    blurKernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}




//cv::Mat ApplyCudaFilter(const cv::Mat& input)
//{
//    cv::Mat output(input.size(), input.type());
//
//    uchar* d_input = nullptr;
//    uchar* d_output = nullptr;
//    size_t size = input.step * input.rows;
//
//    cudaMalloc(&d_input, size);
//    cudaMalloc(&d_output, size);
//
//    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
//
//    dim3 block(16, 16);
//    dim3 grid((input.cols + 15) / 16, (input.rows + 15) / 16);
//    sobelKernel << <grid, block >> > (d_input, d_output, input.cols, input.rows, input.step);
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);
//
//    cudaFree(d_input);
//    cudaFree(d_output);
//
//    return output;
//}
