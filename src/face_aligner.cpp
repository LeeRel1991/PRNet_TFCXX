#include "face_aligner.h"
#include <iostream>
using namespace std;
using namespace cv;

static float kptTemplete[][2]={{38.2946, 51.6963},
                                {73.5318, 51.5014},
                                {56.0252, 71.7366},
                                {41.5493, 92.3655},
                                {70.7299, 92.204}};

FaceAligner::FaceAligner(int num)
{
    if(num!=5)
    {
        std::cout << "unsupport align stragety: kpt = " << num << std::endl;
        num = 5;
    }

    face_template.create(num, 2);
    // TODO 支持5点或更多点的对齐
    for (int i=0; i<num; ++i)
    {
        face_template(i, 0) = kptTemplete[i][0];
        face_template(i, 1) = kptTemplete[i][1];
    }

}

cv::Mat FaceAligner::align_by_kpt(cv::Mat img, cv::Mat_<float> kpt)
{
    Mat aligned_face;
    Mat affine = estimateRigidTransform(kpt, face_template, false);
    if (affine.rows==0 || affine.cols==0)
    {
        cout << "warning: affine matrix empty! \n";
        aligned_face = img.clone();
    }
    else
    {
        warpAffine(img, aligned_face, affine, Size(112, 112));
    }

    return aligned_face;
}
