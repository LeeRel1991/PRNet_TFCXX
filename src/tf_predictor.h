#ifndef TF_PREDICTOR_180602
#define TF_PREDICTOR_180602

#include <string>

#include "image.h"
#include <opencv2/opencv.hpp>
namespace prnet {

class TensorflowPredictor {
public:
  TensorflowPredictor();
  ~TensorflowPredictor();
  void init(int argc, char* argv[]);
  bool load(const std::string& graph_filename, const std::string& inp_layer,
            const std::string& out_layer);
  bool predict(const Image<float>& inp_img, Image<float>& out_img);
  bool predict(const cv::Mat& inp_img, cv::Mat & out_img);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace prnet


#endif /* end of include guard */
