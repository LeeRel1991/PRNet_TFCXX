#ifndef FACE_ALIGNER_H
#define FACE_ALIGNER_H
#include <opencv2/opencv.hpp>


class FaceAligner
{
public:
    FaceAligner(int num=5);

    cv::Mat align_by_kpt(cv::Mat img, cv::Mat_<float> kpt);

private:

    cv::Mat_<float> face_template;

};

#endif // FACE_ALIGNER_H
