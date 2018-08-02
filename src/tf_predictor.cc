#include "tf_predictor.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/config.pb.h"

#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include "simple_timer.h"

using namespace tensorflow;
using namespace cv;
using namespace std;

#define DrawKpt(img, kpt) \
for(int i = 0;i < kpt.rows; i++){ \
     circle(img, Point2d(kpt(i,0), kpt(i,1)), 3, Scalar(255, 255, 0), -1, 8, 0); \
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


namespace prnet {


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.

Status LoadGraph(const string& graph_file,
                 std::unique_ptr<Session>* session,
                 const int gpu_id) {

    /**
     * @note gpuOpts.set_visible_device_list(stream.str())报内存错误，
     * 因此使用graph::SetDefaultDevice
     * @todo 测试二者区别
     */
    std::stringstream stream;
    stream << "/device:GPU:" << gpu_id;
    GraphDef graph_def;
    graph::SetDefaultDevice(stream.str(), &graph_def);

    Status status = ReadBinaryProto(Env::Default(), graph_file, &graph_def);
    if (!status.ok()) {
        return errors::NotFound("Failed to load compute graph at '", graph_file, "'");
    }

    SessionOptions opts;
    GPUOptions *gpuOpts = new GPUOptions;

    gpuOpts->set_per_process_gpu_memory_fraction(0.3);
    gpuOpts->set_allow_growth(true);
    opts.config.set_allocated_gpu_options(gpuOpts);
    //opts.config.set_log_device_placement(true);

    Session* tmp = NewSession(opts);

    session->reset(tmp);
    status = (*session)->Create(graph_def);
    if (!status.ok()) {
        return status;
    }
    return Status::OK();
}

class PRNet::Impl {
public:
//    void init(int argc, char* argv[]) {
//        // We need to call this to set up global state for TensorFlow.
//        tensorflow::port::InitMain(argv[0], &argc, &argv);
//    }

    int load(const std::string& graph_file, const int gpu_id);
    bool predict(const cv::Mat& inp_img, cv::Mat & out_img);

private:
    std::unique_ptr<tensorflow::Session> session;
    std::string input_layer, output_layer;
};

int PRNet::Impl::load(const std::string& graph_file, const int gpu_id)
{

    // First we load and initialize the model.
    Status load_graph_status = LoadGraph(graph_file, &session, gpu_id );
    if (!load_graph_status.ok())
    {
        std::cerr << load_graph_status;
        return -1;
    }

    input_layer = "Placeholder";
    output_layer = "resfcn256/Conv2d_transpose_16/Sigmoid";

    return 0;
}

bool PRNet::Impl::predict(const cv::Mat& inp_img, cv::Mat & out_img)
{
    // Copy from input image
    Eigen::Index inp_w = static_cast<Eigen::Index>(inp_img.cols),
            inp_h = static_cast<Eigen::Index>(inp_img.rows),
            inp_ch = static_cast<Eigen::Index>(inp_img.channels());

    // TODO: No copy
    std::vector<Tensor> output_tensors;
    Tensor input_tensor(DT_FLOAT, {1, inp_h, inp_w, inp_ch});
    memcpy(input_tensor.flat<float>().data(),
           inp_img.data,
           inp_w * inp_h * inp_ch * sizeof(float) );

    // Run
    Status run_status;
    {
        SimpleTimer timer("tf_forward");
        run_status = session->Run({{input_layer, input_tensor}},
                                     {output_layer}, {}, &output_tensors);
    }

    if (!run_status.ok()) {
        std::cerr << "Running model failed: " << run_status;
        return false;
    }

    // Copy to output image
    const Tensor& output_tensor = output_tensors[0];
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();

    assert(tensor.dimension(0) == 1);
    assert(tensor.dimension(3) == 3);
    size_t out_h = static_cast<size_t>(tensor.dimension(1)),
            out_w = static_cast<size_t>(tensor.dimension(2));

    //Warn: 默认3通道彩色图，所以没做条件判断
    out_img.create(out_w, out_h, CV_32FC3);
    const float* data = tensor.data();
    memcpy(out_img.data, data, out_h * out_w * 3 * sizeof(float));
    matUnnormalize(out_img, 256*1.1f);

    return true;
}// PImpl pattern

PRNet::PRNet():
    impl(new Impl())
{

}

PRNet::~PRNet() {}

int PRNet::init(const std::string& graph_file, const std::string& data_dirname, const int gpu_id)
{
    if (!LoadFaceData(data_dirname, &face_data)) {
        return -1;
    }

    return impl->load(graph_file, gpu_id);
}


bool PRNet::predict(const cv::Mat& inp_img, cv::Mat & out_img) {
    return impl->predict(inp_img, out_img);
}


void PRNet::preprocess(const Mat &img, Mat &img_float)
{
    cvtColor(img, img_float, COLOR_BGR2RGB);
    //img_float.convertTo(img_float, CV_32FC3);
    //matNormalize(img_float);
}

Mat_<double> PRNet::getAffineKpt(const Mat &pos_img, int kptNum)
{
    // TODO 0.006ms, need to code improve
    //SimpleTimer timer("getAffineKpt");
    const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
    Mat_<double> kpt68(n_pt, 2);
    for (size_t i = 0; i < n_pt; i++)
    {
        const uint32_t x_idx = face_data.uv_kpt_indices[i];
        const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];

        const int x = int(pos_img.at<Vec3f>(y_idx, x_idx)[0]);
        const int y = int(pos_img.at<Vec3f>(y_idx, x_idx)[1]);

        // Draw circle
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

    Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    matNormalize(img_float);


    for( auto rect:rects)
    {
        Mat face_img = img_float(rect);
        Mat uv_map, aligned_face;
        Mat_<double> spareKpt;
        {
            SimpleTimer timer("impl->predict uv map");
            impl->predict(face_img, uv_map);
        }

        // get five key points  矫正
        {
            SimpleTimer timer("getAffineKpt");
            spareKpt = getAffineKpt(uv_map, 5);
            aligned_face = aligner.align_by_kpt(img(rect), spareKpt);
            //matUnnormalize(aligned_face, 255.f);
            //aligned_face.convertTo(aligned_face, CV_8UC3);
            alignedFaces.push_back(aligned_face);
        }

#ifdef DRAW_IMG
        DrawKpt(face_img, spareKpt);
        imshow("aligned", aligned_face);
        imshow("kpt", face_img);
        waitKey(1);
#endif


    }
}


} // namespace prnet
