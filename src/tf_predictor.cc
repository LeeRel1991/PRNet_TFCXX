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

/* 执行网络forward前检查输入的尺寸，type，channel是否符合要求*/
#define checkImgProp(img) \
{\
    CHECK( img.size() == m_inputGeometry ) << \
        "input image size must be " << m_inputGeometry.width << " x " << m_inputGeometry.height ; \
    CHECK( img.type() == CV_8UC3 ) << "input image type must be CV_8UC3"; \
    CHECK( img.channels() == m_numChannels ) << "input image channel must be " << m_numChannels; \
}

namespace prnet {

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.

Status LoadGraph(const string& graph_file,
                 std::unique_ptr<Session>* sess,
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

    sess->reset(NewSession(opts));
    status = (*sess)->Create(graph_def);
    if (!status.ok()) {
        return status;
    }
    return Status::OK();
}


void FaceCropper::crop(const Mat src, Rect& bbox, Mat &dst)
{

    float old_size = (bbox.width + bbox.height)/2.0;
    float centor_x = bbox.x + bbox.width/2.0;
    float center_y = bbox.y + bbox.height/2.0;

    int dst_size = int(old_size * 1.2);
    dst_size = min(dst_size, src.rows - (int)center_y);
    dst_size = min(dst_size, src.cols - (int)centor_x);

    int dst_x = max((int)(centor_x - dst_size/2.0), 0);
    int dst_y = max((int)(center_y - dst_size/2.0), 0);

    bbox = Rect(dst_x, dst_y, dst_size, dst_size);
    resize(src(bbox), dst, Size(256,256));

}

void FaceCropper::remapLandmarks(Mat1f& arr, Rect src_rect, Rect dst_rect)
{
    int old_x = max(dst_rect.x, 0);
    int old_y = max(dst_rect.y, 0);
    for(int k=0; k< arr.rows; ++k){
        arr(k, 0) = arr(k, 0) * src_rect.width/256 + src_rect.x - old_x;
        arr(k, 1) = arr(k, 1) * src_rect.height/256 + src_rect.y - old_y;
    }

}

class PRNet::Impl {
public:
    Impl();
    int load(const std::string& graph_file, const int gpu_id);
    void forwardNet(const vector<Mat>& imgs, vector<Mat_<float> >& vertices3d);

private:
    std::unique_ptr<tensorflow::Session> m_sess;
    std::string m_inputLayer, m_outputLayer;
    Size m_inputGeometry;
    int m_numChannels;
};

PRNet::Impl::Impl():
    m_inputGeometry(256,256),
    m_numChannels(3)
{

}

int PRNet::Impl::load(const std::string& graph_file, const int gpu_id)
{
    // First we load and initialize the model.
    Status load_graph_status = LoadGraph(graph_file, &m_sess, gpu_id );
    if (!load_graph_status.ok())
    {
        std::cerr << load_graph_status;
        return -1;
    }

    m_inputLayer = "Placeholder";
    m_outputLayer = "resfcn256/Reshape";

    return 0;
}

void PRNet::Impl::forwardNet(const vector<Mat>& imgs, vector<Mat_<float> >& vertices3d)
{
    // preprocess and Copy from input image
    Tensor in_tensor(DT_FLOAT,
                        {imgs.size(),
                         m_inputGeometry.height,
                         m_inputGeometry.width,
                         m_numChannels});

    size_t data_len = m_inputGeometry.height * m_inputGeometry.width * m_numChannels ;
    auto inp_data = in_tensor.flat<float>().data();
    for (auto img:imgs) {
#ifdef ENABLE_CHECK
        checkImgProp(img);
#endif

        Mat img_float;
        img.convertTo(img_float, CV_32FC3);
        matNormalize(img_float, 255.f);
        memcpy(inp_data, img_float.data,  data_len * sizeof(float) );
        inp_data += data_len;
    }

    // Run
    std::vector<Tensor> outputs;
    Status status = m_sess->Run({{m_inputLayer, in_tensor}},
                                     {m_outputLayer}, {}, &outputs);

    TF_CHECK_OK(status);

    // Copy to output image
    const Tensor& out_tensor = outputs[0];
    TTypes<float, 4>::ConstTensor tensor = out_tensor.tensor<float, 4>();

    CHECK_EQ(tensor.dimension(0), imgs.size()) << "output must have same batch with input";
    CHECK_EQ(tensor.dimension(3), 1) << "outpur channel must be 1";
    size_t out_h = static_cast<size_t>(tensor.dimension(1)),
            out_w = static_cast<size_t>(tensor.dimension(2));

    //Warn: 输出65536x3点云矩阵
    size_t out_len = out_h * out_w;
    for(int i =0; i<imgs.size(); ++i)
    {
        Mat_<float> tmp(out_h, out_w);
        memcpy(tmp.data, tensor.data() + i * data_len, out_len * sizeof(float) );
        vertices3d.push_back(tmp);
    }

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


void PRNet::predict(const vector<Mat>& imgs, vector<cv::Mat1f >& vertices3d) {
    impl->forwardNet(imgs, vertices3d);
}

void PRNet::predict(const std::vector<Mat> &imgs,
                    const std::vector<std::vector<Rect> >& rects,
                    std::vector<std::vector<Mat1f> >& landmarks)
{
    // 准备batch input
    vector<Mat> imgs_batch;
    std::vector<std::vector<Rect> > new_rects = rects;
    auto it_img = imgs.cbegin(), it_img_end = imgs.cend();
    auto it_rect = new_rects.begin();
    for(; it_img!=it_img_end; ++it_img, ++it_rect){
        for(auto it = it_rect->begin(); it!= it_rect->end(); ++it){
            Mat face_img;
            cropper.crop(*it_img, *it, face_img);
            imgs_batch.push_back(face_img);
        }
    }

    // forward
    vector<Mat_<float> > vertices3d_batch;
    {
        SimpleTimer timer("impl->predict uv map");
        impl->forwardNet(imgs_batch, vertices3d_batch);
    }

    // output remap to original boundingbox
    SimpleTimer timer("remap output landmarks");
    landmarks.resize(imgs_batch.size());
    auto it_kpt = vertices3d_batch.cbegin();
    for(int i=0; i<imgs.size(); ++i){
        int bbox_cnt = rects[i].size();

        for(int j=0; j< bbox_cnt; ++j){
            Mat1f kpt = getAffineKpt(*it_kpt, 5);
            cropper.remapLandmarks(kpt, new_rects[i][j], rects[i][j]);

            landmarks[i].push_back(kpt);
            it_kpt++;

#ifdef DRAW_IMG
            Mat face_img = imgs[i](rects[i][j]);
            DrawKpt(face_img, kpt);
            char tmp[10];
            sprintf(tmp, "kpt_%d", i);
            imshow(tmp, face_img);
            waitKey(1);
#endif
        }
    }
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
    vector<Mat> imgs_batch;
    vector<Mat_<float> > vertices3d_batch;
    for( auto rect:rects)
    {
        Mat face_img;
        cropper.crop(img, rect, face_img);
        imgs_batch.push_back(face_img);
    }

    {
        SimpleTimer timer("impl->predict uv map");
        impl->forwardNet(imgs_batch, vertices3d_batch);
    }

    for(int i=0; i< imgs_batch.size(); ++i)
    {
        Mat face_img = imgs_batch[i];
        Mat aligned_face;
        Mat_<float> spareKpt, vertices3d=vertices3d_batch[i];

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
        char tmp[10];
        sprintf(tmp, "kpt_%d", i);
        imshow(tmp, face_img);
        waitKey(1);
#endif
    }

}


} // namespace prnet
