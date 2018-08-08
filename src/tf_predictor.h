#ifndef TF_PREDICTOR_180602
#define TF_PREDICTOR_180602

#include <string>

#include "face-data.h"
#include "face_aligner.h"
#include <opencv2/opencv.hpp>
namespace prnet {

class FaceCropper{

public:
    void crop(const cv::Mat src, cv::Rect& bbox, cv::Mat& dst);
    void remapLandmarks(cv::Mat1f& arr,  cv::Rect cropped_rect, cv::Rect old_rect);

};


class PRNet {
public:
    PRNet();
    ~PRNet();
    /**
     * @brief init
     * @param graph_file
     * @param data_dirname
     * @param gpu_id
     * @return
     */
    int init(const std::string& graph_file, const std::string& data_dirname, const int gpu_id);
    bool load(const std::string& graph_file);
    void predict(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat1f >& vertices3d);
    /**
     * @brief predict 人脸关键点检测
     * @param imgs input 批次图像，大图
     * @param rects input 批次boundingbox
     * @param landmarks output 关键点，坐标相对于@em rects
     */
    void predict(const std::vector<cv::Mat> &imgs,
                 const std::vector<std::vector<cv::Rect> >& rects,
                 std::vector<std::vector<cv::Mat1f > > &landmarks);

    /**
     * @brief align
     * @param img
     * @param rects
     * @param alignedFaces
     */
    void align(const cv::Mat& img, const std::vector<cv::Rect>& rects, std::vector<cv::Mat>& alignedFaces);

    /**
     * @brief preprocess 对原始图片按照进行预处理，包括转为CV_32FC3
     * @param img       input
     * @param img_float output
     */
    static void preprocess(const cv::Mat& img, cv::Mat& img_float);


private:
    cv::Mat_<float> getAffineKpt(const cv::Mat &pos_img, int kptNum=5);

    class Impl;
    FaceAligner aligner;
    FaceData face_data;
    FaceCropper cropper;
    std::unique_ptr<Impl> impl;
};

} // namespace prnet


#endif /* end of include guard */
