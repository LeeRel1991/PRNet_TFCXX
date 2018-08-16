#include "utils.h"

void matNormalize(cv::Mat& img, float scale){
    int nr = img.rows;
    int nc = img.cols * img.channels();
    for(int i = 0; i < nr; ++i)
    {
        float* data = img.ptr<float>(i);
        for(int j= 0; j< nc; ++j)
        {
            data[j]=  data[j] * scale;
        }
    }
}


void matNormalize(cv::cuda::GpuMat& img, float scale){
    cv::cuda::GpuMat tmp(img.rows, img.cols, CV_32FC3, cv::Scalar(scale,scale,scale));
    cv::cuda::divide(img, tmp, img);
}


