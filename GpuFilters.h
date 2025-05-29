#pragma once

#include <opencv2/core.hpp>

// CUDA filter declarations
cv::Mat ApplyCudaInvert(const cv::Mat& input);   // Box or Gaussian blur
cv::Mat ApplyCudaFilterEdge(const cv::Mat& input); // Sobel or edge filter
cv::Mat ApplyCudaBlur(const cv::Mat& input);   // Box or Gaussian blur

