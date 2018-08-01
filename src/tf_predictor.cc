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


using namespace tensorflow;
using namespace cv;
using namespace std;

#define DrawKpt(img, kpt) \
for(int i = 0;i < kpt.rows; i++){ \
     circle(img, Point2d(kpt(i,0), kpt(i,1)),3, Scalar(255, 255, 0), -1, 8, 0); \
}


#define matNormalize(img) \
{\
    int nr = img.rows; \
    int nc = img.cols * img.channels(); \
    for(int i = 0; i < nr; ++i) \
    { \
        float* data = img.ptr<float>(i); \
        for(int j= 0; j< nc; ++j) \
        { \
            data[j]=  data[j]/255.f; \
        } \
    }\
}

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


static double kptTemplete[][2]={{38.2946, 51.6963},
                                {73.5318, 51.5014},
                                {56.0252, 71.7366},
                                {41.5493, 92.3655},
                                {70.7299, 92.204}};


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

class PRNet::Impl {
public:
//    void init(int argc, char* argv[]) {
//        // We need to call this to set up global state for TensorFlow.
//        tensorflow::port::InitMain(argv[0], &argc, &argv);
//    }

    int load(const std::string& graph_filename,
             const std::string& inp_layer,
             const std::string& out_layer);

    bool predict(const cv::Mat& inp_img, cv::Mat & out_img);

    bool predict(const Image<float>& inp_img, Image<float>& out_img);

private:
    std::unique_ptr<tensorflow::Session> session;
    std::string input_layer, output_layer;
};

int PRNet::Impl::load(const std::string& graph_filename,
         const std::string& inp_layer,
         const std::string& out_layer)
{

    // First we load and initialize the model.
    Status load_graph_status = LoadGraph(graph_filename, &session);
    if (!load_graph_status.ok())
    {
        std::cerr << load_graph_status;
        return -1;
    }

    input_layer = inp_layer;
    output_layer = out_layer;

    return 0;
}

bool PRNet::Impl::predict(const cv::Mat& inp_img, cv::Mat & out_img)
{
    // Copy from input image
    Eigen::Index inp_width = static_cast<Eigen::Index>(inp_img.cols),
            inp_height = static_cast<Eigen::Index>(inp_img.rows),
            inp_channels = static_cast<Eigen::Index>(inp_img.channels());

    // TODO: No copy
    std::vector<Tensor> output_tensors;
    Tensor input_tensor(DT_FLOAT, {1, inp_height, inp_width, inp_channels});
    memcpy(input_tensor.flat<float>().data(), inp_img.data, inp_width * inp_height * inp_channels * sizeof(float) );

    auto startT = std::chrono::system_clock::now();

    // Run
    Status run_status = session->Run({{input_layer, input_tensor}},
                                 {output_layer}, {}, &output_tensors);

    auto endT = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> ms = endT - startT;
    std::cout << "Ran forward. elapsed = " << ms.count() << " [ms] " << std::endl;
    if (!run_status.ok()) {
        std::cerr << "Running model failed: " << run_status;
        return false;
    }

    // Copy to output image
    const Tensor& output_tensor = output_tensors[0];
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();

    assert(tensor.dimension(0) == 1);
    assert(tensor.dimension(3) == 3);
    size_t out_height = static_cast<size_t>(tensor.dimension(1)),
            out_width = static_cast<size_t>(tensor.dimension(2));

    //Warn: 默认3通道彩色图，所以没做条件判断
    out_img.create(out_width, out_height, CV_32FC3);
    const float* data = tensor.data();
    memcpy(out_img.data, data, out_height * out_width * 3 * sizeof(float));
    matUnnormalize(out_img, 256*1.1f);

    return true;
}

bool PRNet::Impl::predict(const Image<float>& inp_img, Image<float>& out_img)
{
    // Copy from input image
    Eigen::Index inp_width = static_cast<Eigen::Index>(inp_img.getWidth()),
            inp_height = static_cast<Eigen::Index>(inp_img.getHeight()),
            inp_channels = static_cast<Eigen::Index>(inp_img.getChannels());
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

    out_img.foreach([&](int x, int y, int c, float& v)
    {
        v = tensor(0, y, x, c);
    });

    return true;
}

// PImpl pattern


PRNet::PRNet():
    impl(new Impl())
{

}

PRNet::~PRNet() {}

int PRNet::init(const std::string& graph_filename, const std::string& data_dirname)
{
    if (!LoadFaceData(data_dirname, &face_data)) {
        return -1;
    }

    return impl->load(graph_filename, "Placeholder", "resfcn256/Conv2d_transpose_16/Sigmoid");
}


bool PRNet::predict(const cv::Mat& inp_img, cv::Mat & out_img) {
    return impl->predict(inp_img, out_img);
}

bool PRNet::predict(const Image<float>& inp_img, Image<float>& out_img) {
    return impl->predict(inp_img, out_img);
}

void PRNet::preprocess(const Mat &img, Mat &img_float)
{
    cvtColor(img, img_float, COLOR_BGR2RGB);
    img_float.convertTo(img_float, CV_32FC3);

    matNormalize(img_float);
}

Mat_<double> PRNet::getAffineKpt(const Mat &pos_img, int kptNum)
{
    const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
    Mat_<double> kpt68(n_pt, 2);
    for (size_t i = 0; i < n_pt; i++)
    {
        const uint32_t x_idx = face_data.uv_kpt_indices[i];
        const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];

        const int x = int(pos_img.at<Vec3f>(y_idx, x_idx)[0]);
        const int y = int(pos_img.at<Vec3f>(y_idx, x_idx)[1]);

        // Draw circle
//        circle(img, Point(x, y), 3,  Scalar(255,0,0), -1, 0, 0);
        kpt68(i, 0) = x;
        kpt68(i, 1) = y;
    }


    Mat_<double> sparseKpt(kptNum, 2);

    switch (kptNum) {
    case 5:
        sparseKpt(2, 0) = kpt68(30, 0);
        sparseKpt(2, 1) = kpt68(30, 1);
        sparseKpt(3, 0) = kpt68(48, 0);
        sparseKpt(3, 1) = kpt68(48, 1);
        sparseKpt(4, 0) = kpt68(54, 0);
        sparseKpt(4, 1) = kpt68(54, 1);

        sparseKpt(0, 0) = (kpt68(36, 0) + kpt68(39, 0) ) /2;
        sparseKpt(0, 1) = (kpt68(37, 1) + kpt68(38, 1) + kpt68(40, 1) + kpt68(41, 1) ) /4;
        sparseKpt(1, 0) = (kpt68(42, 0) + kpt68(45, 0) ) /2;
        sparseKpt(1, 1) = (kpt68(43, 1) + kpt68(44, 1) + kpt68(46, 1) + kpt68(47, 1) ) /4;
        break;
    default:
        break;
    }

    return sparseKpt;
}

void PRNet::align(const Mat &img, const std::vector<Rect> &rects, std::vector<Mat> &alignedFaces)
{

    for( auto rect:rects)
    {
        auto startT = std::chrono::system_clock::now();
        Mat face_img = img(rect).clone();
        Mat uv_map;

        // TODO forward 前后时间优化
        impl->predict(face_img, uv_map);

        auto endT = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> ms = endT - startT;
        std::cout << "predict. elapsed = " << ms.count() << " [ms] " << std::endl;

        // get five key points
        //矫正
        startT = std::chrono::system_clock::now();
        Mat_<double> spareKpt = getAffineKpt(uv_map, 5);

        endT = std::chrono::system_clock::now();
        ms = endT - startT;
        std::cout << "getAffineKpt. elapsed = " << ms.count() << " [ms] " << std::endl;

        Mat aligned_face;

        startT = std::chrono::system_clock::now();
        Mat_<double> alignTemplete(5, 2);
        for (int i=0; i<5; ++i)
        {
            alignTemplete(i, 0) = kptTemplete[i][0];
            alignTemplete(i, 1) = kptTemplete[i][1];
        }

        Mat affine = estimateRigidTransform(spareKpt, alignTemplete, false);
        if (affine.rows==0 || affine.cols==0)
            cout << "warning: affine matrix empty! \n";
        else
        {

            warpAffine(face_img, aligned_face, affine, Size(112, 112));

            startT = std::chrono::system_clock::now();
            matUnnormalize(aligned_face, 255.f);
            endT = std::chrono::system_clock::now();
            ms = endT - startT;
            std::cout << "align face. elapsed = " << ms.count() << " [ms] " << std::endl;
            aligned_face.convertTo(aligned_face, CV_8UC3);
//            imshow("aligned", aligned_face);
        }
        alignedFaces.push_back(aligned_face);


#ifdef DRAW_IMG
        DrawKpt(face_img, spareKpt);
        imshow("kpt", face_img);
        waitKey(1);
#endif


    }
}


} // namespace prnet
