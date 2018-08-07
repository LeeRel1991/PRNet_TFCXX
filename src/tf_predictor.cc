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
#include "utils.h"
using namespace tensorflow;
using namespace cv;
using namespace std;

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


void FaceCropper::crop(const Mat src, Rect bbox, Mat &dst)
{

    float old_size = (bbox.width + bbox.height)/2.0;
    float centor_x = bbox.x + bbox.width/2.0;
    float center_y = bbox.y + bbox.height/2.0;

    int dst_size = int(old_size * 1.2);
    dst_size = min(dst_size, src.rows - (int)center_y);
    dst_size = min(dst_size, src.cols - (int)centor_x);

    int dst_x = max((int)(centor_x - dst_size/2.0), 0);
    int dst_y = max((int)(center_y - dst_size/2.0), 0);

    Rect dst_bbox(dst_x, dst_y, dst_size, dst_size);
    resize(src(dst_bbox), dst, Size(256,256));

}

class PRNet::Impl {
public:
    int load(const std::string& graph_file, const int gpu_id);
    bool predict(const cv::Mat& inp_img, cv::Mat_<float>& out_img);

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
    output_layer = "resfcn256/Reshape";
//    output_layer = "resfcn256/Conv2d_transpose_16/Sigmoid";

    return 0;
}

bool PRNet::Impl::predict(const cv::Mat& inp_img, cv::Mat_<float> & out_img)
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
    Status run_status = session->Run({{input_layer, input_tensor}},
                                     {output_layer}, {}, &output_tensors);

    if (!run_status.ok()) {
        std::cerr << "Running model failed: " << run_status;
        return false;
    }

    // Copy to output image
    const Tensor& output_tensor = output_tensors[0];
    TTypes<float, 4>::ConstTensor tensor = output_tensor.tensor<float, 4>();

    assert(tensor.dimension(0) == 1);
    assert(tensor.dimension(3) == 1); //batch x 65536x3x1
    size_t out_h = static_cast<size_t>(tensor.dimension(1)),
            out_w = static_cast<size_t>(tensor.dimension(2));

    //Warn: 输出65536x3点云矩阵
    out_img.create(out_h, out_w);
    memcpy(out_img.data, tensor.data(), out_h * out_w * sizeof(float));


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


bool PRNet::predict(const cv::Mat& inp_img, cv::Mat_<float> & out_img) {
    return impl->predict(inp_img, out_img);
}


void PRNet::preprocess(const Mat &img, Mat &img_float)
{
    cvtColor(img, img_float, COLOR_BGR2RGB);
}

Mat_<float> PRNet::getAffineKpt(const Mat &pos_img, int kptNum)
{
    // Note 0.006ms total
    //SimpleTimer timer("getAffineKpt00");
    const size_t n_pt = face_data.uv_kpt_indices.size();
    Mat_<float> kpt68(n_pt, 2);

    float* data = (float*)pos_img.data;
    for (size_t i = 0; i < n_pt; i++)
    {
        const uint32_t idx = face_data.uv_kpt_indices[i];
        kpt68(i, 0) =  *(data + idx *3 + 0);
        kpt68(i, 1) =  *(data + idx *3 + 1);
    }


    Mat_<float> sparseKpt(kptNum, 2);
    int data_len = 2*sizeof(float);
    switch (kptNum) {
    case 5:
        memcpy((float*)sparseKpt.data + 2 * 2, (float*)kpt68.data + 30*2, data_len);
        memcpy((float*)sparseKpt.data + 3 * 2, (float*)kpt68.data + 48*2, data_len);
        memcpy((float*)sparseKpt.data + 4 * 2, (float*)kpt68.data + 54*2, data_len);

        sparseKpt(0, 0) = (kpt68(36, 0) + kpt68(39, 0) ) /2;
        sparseKpt(0, 1) = (kpt68(37, 1) + kpt68(38, 1) + kpt68(40, 1) + kpt68(41, 1) ) /4;
        sparseKpt(1, 0) = (kpt68(42, 0) + kpt68(45, 0) ) /2;
        sparseKpt(1, 1) = (kpt68(43, 1) + kpt68(44, 1) + kpt68(46, 1) + kpt68(47, 1) ) /4;
        break;

    default:
        sparseKpt = kpt68.clone();
        break;
    }

    return sparseKpt;
}

void PRNet::align(const Mat &img, const std::vector<Rect> &rects, std::vector<Mat> &alignedFaces)
{
    for( auto rect:rects)
    {
        Mat face_img, face_float;
        cropper.crop(img, rect, face_img);
        face_img.convertTo(face_float, CV_32FC3);
        matNormalize(face_float, 255.f);

        Mat aligned_face;
        Mat_<float> spareKpt, vertices3d;
        {
            SimpleTimer timer("impl->predict uv map");
            impl->predict(face_float, vertices3d);
        }
        if(vertices3d.rows==0 || vertices3d.cols ==0)
        {
            aligned_face = face_img.clone();
            continue;
        }
        // get five key points  矫正
        {
            SimpleTimer timer("getAffineKpt");
            spareKpt = getAffineKpt(vertices3d, 5);
            aligned_face = aligner.align_by_kpt(face_img, spareKpt);

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
