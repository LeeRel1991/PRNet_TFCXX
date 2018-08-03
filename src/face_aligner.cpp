#include "face_aligner.h"
#include "utils.h"
#include <iostream>
using namespace std;
using namespace cv;

static float kptTemplete[][2]={{38.2946/112, 51.6963/112},
                                {73.5318/112, 51.5014/112},
                                {56.0252/112, 71.7366/112},
                                {41.5493/112, 92.3655/112},
                                {70.7299/112, 92.204/112}};

FaceAligner::FaceAligner(int num)
{
    if(num!=5)
    {
        std::cout << "unsupport align stragety: kpt = " << num << std::endl;
        num = 5;
    }

    // TODO 支持5点或更多点的对齐
    face_template = Mat_<float>(num, 2, (float*)kptTemplete);

}

cv::Mat FaceAligner::align_by_kpt(cv::Mat img, cv::Mat_<float> kpt)
{
    Mat aligned_face;
    Mat tmp = face_template.clone();
    matUnnormalize(tmp, img.rows);
    Mat affine = estimateRigidTransform(kpt, tmp, false);
    if (affine.rows==0 || affine.cols==0)
    {
        cout << "warning: affine matrix empty! \n";
        aligned_face = img.clone();
    }
    else
    {
        warpAffine(img, aligned_face, affine, Size(img.rows, img.cols));
    }

    return aligned_face;
}
