#ifndef TF_PREDICTOR_180602
#define TF_PREDICTOR_180602

#include <string>

#include "face-data.h"
#include "face_aligner.h"
#include <opencv2/opencv.hpp>
namespace prnet {

class FaceCropper{

public:
    void crop(const cv::Mat src, cv::Rect bbox, cv::Mat& dst);

};


class PRNet {
public:
    PRNet();
    ~PRNet();
    int init(const std::string& graph_file, const std::string& data_dirname, const int gpu_id);
    bool load(const std::string& graph_file);
    bool predict(const cv::Mat& inp_img, cv::Mat & out_img);

    /**
     * @brief align
     * @param img
     * @param rects
     * @param alignedFaces
     */
    void align(const cv::Mat& img, const std::vector<cv::Rect>& rects, std::vector<cv::Mat>& alignedFaces);

    /**
     * @brief preprocess 对原始图片按照进行预处理，包括转为CV_32FC3，像素值归一化
     * @param img       input
     * @param img_float output
     */
    static void preprocess(const cv::Mat& img, cv::Mat& img_float);


private:
    cv::Mat_<double> getAffineKpt(const cv::Mat &pos_img, int kptNum=5);

    class Impl;
    FaceAligner aligner;
    FaceData face_data;
    FaceCropper cropper;
    std::unique_ptr<Impl> impl;
};

} // namespace prnet


#endif /* end of include guard */
