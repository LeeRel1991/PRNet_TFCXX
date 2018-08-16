#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
//---------------------CUDA头文件------
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_device_runtime_api.h>
//---------------------CUDA头文件------


// 画点
#define DrawKpt(img, kpt) \
for(int i = 0;i < kpt.rows; i++){ \
     circle(img, Point2d(kpt(i,0), kpt(i,1)), 3, Scalar(255, 255, 0), -1, 8, 0); \
}

//画矩形框
#define drawBoundingbox(frame, rects) \
    for(auto r:rects) \
        cv::rectangle(frame, r, cv::Scalar(0,255,0) , 2);


//归一化，每个像素值除以scale
void matNormalize(cv::Mat& img, float scale);
void matNormalize(cv::cuda::GpuMat& img, float scale);



//去归一化，每个像素乘以scale
#define matUnnormalize(img,scale) \
{\
    int nr = img.rows; \
    int nc = img.cols * img.channels(); \
    for(int i = 0; i < nr; ++i) \
    { \
        float* data = img.ptr<float>(i); \
        for(int j= 0; j< nc; ++j) \
        { \
            data[j]=  data[j]*scale; \
        } \
    }\
}

#endif // UTILS_H
