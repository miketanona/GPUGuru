#pragma once

#include <opencv2/core.hpp>

// CUDA filter declarations
cv::Mat ApplyCudaFilter(const cv::Mat& input); // Sobel or edge filter
cv::Mat ApplyCudaBlur(const cv::Mat& input);   // Box or Gaussian blur