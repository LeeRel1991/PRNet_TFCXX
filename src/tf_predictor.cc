#include "tf_predictor.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

using namespace tensorflow;
using namespace cv;
using namespace std;

namespace prnet {

namespace {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {

    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
      return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                          graph_file_name, "'");
    }

    tensorflow::SessionOptions opts;

    GPUOptions *gpuOpts = new GPUOptions;
    int gpu_id = 0;
    std::stringstream stream;
    stream << gpu_id;

    gpuOpts->set_per_process_gpu_memory_fraction(0.3);
    gpuOpts->set_allow_growth(true);
//    gpuOpts.set_visible_device_list(stream.str());
    opts.config.set_allocated_gpu_options(gpuOpts);
    opts.config.set_log_device_placement(true);

    Session* tmp = tensorflow::NewSession(opts);

  session->reset(tmp);
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

} // anonymous namespace

class TensorflowPredictor::Impl {
public:
  void init(int argc, char* argv[]) {
    // We need to call this to set up global state for TensorFlow.
    tensorflow::port::InitMain(argv[0], &argc, &argv);
  }

  bool load(const std::string& graph_filename, const std::string& inp_layer,
            const std::string& out_layer) {
    // First we load and initialize the model.
    Status load_graph_status = LoadGraph(graph_filename, &session);
    if (!load_graph_status.ok()) {
      std::cerr << load_graph_status;
      return false;
    }

    input_layer = inp_layer;
    output_layer = out_layer;

    return true;
  }

  bool predict(const cv::Mat& inp_img, cv::Mat & out_img)
  {
      // Copy from input image
    Eigen::Index inp_width = static_cast<Eigen::Index>(inp_img.cols);
    Eigen::Index inp_height = static_cast<Eigen::Index>(inp_img.rows);
    Eigen::Index inp_channels = static_cast<Eigen::Index>(inp_img.channels());
    Tensor input_tensor(DT_FLOAT, {1, inp_height, inp_width, inp_channels});

    // TODO: No copy
    memcpy(input_tensor.flat<float>().data(), inp_img.data, inp_width * inp_height * inp_channels * sizeof(float) );

    auto startT = std::chrono::system_clock::now();

    // Run
    std::vector<Tensor> output_tensors;
    Status run_status = session->Run({{input_layer, input_tensor}},
                                     {output_layer}, {}, &output_tensors);

    auto endT = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> ms = endT - startT;
    std::cout << "Ran forward. elapsed = " << ms.count() << " [ms] " << std::endl;
    if (!run_status.ok()) {
      std::cerr << "Running model failed: " << run_status;
      return false;
    }
    const Tensor& output_tensor = output_tensors[0];

    // Copy to output image
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();
    assert(tensor.dimension(0) == 1);
    size_t out_height = static_cast<size_t>(tensor.dimension(1));
    size_t out_width = static_cast<size_t>(tensor.dimension(2));
    size_t out_channels = static_cast<size_t>(tensor.dimension(3));
    out_img.create(out_width, out_height, CV_32FC3);

    //TODO: 测试mat访问像素赋值的高效方法
    for(int i = 0; i < out_height; ++i)
    {
        for(int j= 0; j< out_width; ++j)
        {
            out_img.at<cv::Vec3f>(i, j)[0] = tensor(0, i, j, 0) * 256 * 1.1;
            out_img.at<cv::Vec3f>(i, j)[1] = tensor(0, i, j, 1) * 256 * 1.1;
            out_img.at<cv::Vec3f>(i, j)[2] = tensor(0, i, j, 2) * 256 * 1.1;
        }
    }

    return true;
  }

  bool predict(const Image<float>& inp_img, Image<float>& out_img) {
    // Copy from input image
    Eigen::Index inp_width = static_cast<Eigen::Index>(inp_img.getWidth());
    Eigen::Index inp_height = static_cast<Eigen::Index>(inp_img.getHeight());
    Eigen::Index inp_channels = static_cast<Eigen::Index>(inp_img.getChannels());
    Tensor input_tensor(DT_FLOAT, {1, inp_height, inp_width, inp_channels});
    // TODO: No copy
    std::copy_n(inp_img.getData(), inp_width * inp_height * inp_channels,
                input_tensor.flat<float>().data());

    auto startT = std::chrono::system_clock::now();

    // Run
    std::vector<Tensor> output_tensors;
    Status run_status = session->Run({{input_layer, input_tensor}},
                                     {output_layer}, {}, &output_tensors);

    auto endT = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> ms = endT - startT;
    std::cout << "Image Ran forward. elapsed = " << ms.count() << " [ms] " << std::endl;

    if (!run_status.ok()) {
      std::cerr << "Running model failed: " << run_status;
      return false;
    }
    const Tensor& output_tensor = output_tensors[0];

    // Copy to output image
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();
    assert(tensor.dimension(0) == 1);
    size_t out_height = static_cast<size_t>(tensor.dimension(1));
    size_t out_width = static_cast<size_t>(tensor.dimension(2));
    size_t out_channels = static_cast<size_t>(tensor.dimension(3));
    out_img.create(out_width, out_height, out_channels);
    out_img.foreach([&](int x, int y, int c, float& v) {
      v = tensor(0, y, x, c);
    });


    return true;
  }

private:
  std::unique_ptr<tensorflow::Session> session;
  std::string input_layer, output_layer;
};

// PImpl pattern
TensorflowPredictor::TensorflowPredictor() : impl(new Impl()) {}
TensorflowPredictor::~TensorflowPredictor() {}
void TensorflowPredictor::init(int argc, char* argv[]) {
  impl->init(argc, argv);
}
bool TensorflowPredictor::load(const std::string& graph_filename,
                               const std::string& inp_layer,
                               const std::string& out_layer) {
  return impl->load(graph_filename, inp_layer, out_layer);
}
bool TensorflowPredictor::predict(const cv::Mat& inp_img, cv::Mat & out_img) {
  return impl->predict(inp_img, out_img);
}

bool TensorflowPredictor::predict(const Image<float>& inp_img,
                                  Image<float>& out_img) {
  return impl->predict(inp_img, out_img);
}

} // namespace prnet
