#ifndef TF_PREDICTOR_180602
#define TF_PREDICTOR_180602

#include <string>

#include "face-data.h"
#include "face_aligner.h"
#include <opencv2/opencv.hpp>
namespace prnet {

class FaceCropper{

public:
    void crop(const cv::cuda::GpuMat src, cv::Rect& bbox, cv::cuda::GpuMat& dst);
    void remapLandmarks(cv::Mat1f& arr,  cv::Rect cropped_rect, cv::Rect old_rect);

};


class PRNet {
public:
    PRNet();
    ~PRNet();

    /** initialize， must be called before predict
     * @brief init
     * @param graph_file  .pb文件
     * @param data_dirname uv-data 路径， see in ./data/uv-data
     * @param lmk_num, number of keypoints, 5 or 68, default 5
     * @param gpu_id, gpu device id, 0, 1 ..
     * @return
     */
    int init(const std::string& graph_file,
             const std::string& data_dirname,
             const int lmk_num=5,
             const int gpu_id=0);

    /**
     * @brief predict output 3D point clouds for a rgb image
     * @param imgs input rgb image (256x256)
     * @param vertices3d output 3D point coord. 65536x3
     */
    void predict(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat1f >& vertices3d);

    /**
     * @brief predict 人脸关键点检测
     * @param imgs input 批次图像，大图, cv::Mat on cpu
     * @param rects input 批次boundingbox
     * @param landmarks output 关键点，坐标相对于@em rects,
     * @note @em landmarks size : numx2, cv::Mat1f: Mat_<float>， usage is similar to Mat
     * 在图像上画点:
     * cv::Mat1f kpt(5, 2);
     * cv::Mat img;
     * for(int i = 0;i < kpt.rows; i++){
     *    circle(img, Point2d(kpt(i,0), kpt(i,1)), 3, Scalar(255, 255, 0), -1, 8, 0);
     *  }
     *
     * 访问某个点坐标：
     *  cv::Point2f p;
     *  p.x = kpt(i, 0);
     *  p.y = kpt(i, 1);
     */
    void predict(const std::vector<cv::Mat> &imgs,
                 const std::vector<std::vector<cv::Rect> >& rects,
                 std::vector<std::vector<cv::Mat1f > > &landmarks);

    /** @overload
     * @brief predict
     * @param imgs input 批次图像，大图, cv::cuda::GpuMat on gpu
     * @param rects
     * @param landmarks
     */
    void predict(const std::vector<cv::cuda::GpuMat> &imgs,
                 const std::vector<std::vector<cv::Rect> >& rects,
                 std::vector<std::vector<cv::Mat1f > > &landmarks);

    /** demo for align face img use 5 landmarks
     * @brief align
     * @param img input image to align
     * @param predLmk input landmarks obtained by prnet
     * @param alignedFace output 尺寸与img相同
     */
    void align(const cv::Mat& img, const cv::Mat1f predLmk, cv::Mat & alignedFace);

    /**
     * @brief preprocess 对原始图片按照进行预处理，BGR->RGB
     * @param img       input
     * @param img_float output
     */
     static void preprocess(const cv::Mat& img, cv::Mat& img_rgb);


private:
    cv::Mat_<float> getAffineKpt(const cv::Mat &pos_img);

    int             m_lmkNum;
    class Impl;
    FaceAligner aligner;
    FaceData face_data;
    FaceCropper cropper;
    std::unique_ptr<Impl> impl;
};

} // namespace prnet


#endif /* end of include guard */
