
#include "cxxopts.hpp"

#include "tf_predictor.h"
#include "simple_timer.h"
#include "utils.h"
#include "mxnet_ssd_classifier.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tensorflow/core/platform/init_main.h>
#include <opencv2/opencv.hpp>
#include <fstream>
using namespace cv;
using namespace prnet;
using namespace std;

int parseArgvParams(int argc, char **argv,
                    std::string& image_filename,
                    std::string& graph_filename,
                    std::string& data_dirname)
{
    cxxopts::Options options("prnet-infer", "PRNet infererence in C++");
    options.add_options()("i,image", "Input image file",
                          cxxopts::value<std::string>())(
        "g,graph", "Input freezed graph file", cxxopts::value<std::string>())(
        "d,data", "Data folder of PRNet repo", cxxopts::value<std::string>());

    auto result = options.parse(argc, argv);

    if (!result.count("image")) {
      std::cerr << "Please specify input image with -i or --image option."
                << std::endl;
      return -1;
    }

    if (!result.count("graph")) {
      std::cerr << "Please specify freezed graph with -g or --graph option."
                << std::endl;
      return -1;
    }

    if (!result.count("data")) {
      std::cerr
          << "Please specify Data folder of PRNet repo with -d or --data option."
          << std::endl;
      return -1;
    }

    image_filename = result["image"].as<std::string>();
    graph_filename = result["graph"].as<std::string>();
    data_dirname = result["data"].as<std::string>();

    return 0;
}

void getFaceBoungingbox(MxnetSSDClassifier *Classifier, const Mat& frame, std::vector<Rect>& faces)
{
    SimpleTimer timer("detect face using mobileSSD");
    ImageData img;
    img.vCpuImg.push_back(frame);
    std::vector<std::vector<classifyResult> > Results;
    Results = Classifier->classifier(img); //目标检测,同时保存每个框的置信度

//    for (auto &item : Results){
          for (auto& bbox :Results[0]) {
              int x = (int)(bbox.x);
              int y = (int)(bbox.y);
              int w = (int)(bbox.w);
              int h = (int)(bbox.h);
              faces.push_back(cv::Rect(x, y, w , h));

          }
//      }
}

int main(int argc, char **argv)
{
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string image_filename, pb_model, uv_files;
    if(parseArgvParams(argc, argv, image_filename, pb_model, uv_files) != 0)
        return -1;

    // Load image
    std::cout << "Loading image \"" << image_filename << "\"" << std::endl;

    // -- 1. Load the face detector cascades
    MxnetSSDClassifier *detector = new MxnetSSDClassifier("../../third_party/mobileSSD_MX/model/deploy_ssd_mobilenet_v2_300-symbol.json",
                                                            "../../third_party/mobileSSD_MX/model/deploy_ssd_mobilenet_v2_300-0100.params",
                                                            "../../third_party/mobileSSD_MX/model/label.txt",true,1);



    Mat frame;
    VideoCapture capture;
    capture.open(image_filename);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    // Predict
    PRNet tf_predictor;
    if(0!=tf_predictor.init(pb_model, uv_files, 0))
    {
        std::cout << "Initialized tf model fails" << std::endl;
        return -1;
    }

    for(;;)
    {
        capture >> frame ;
        if (frame.empty())
            break;

        std::cout << "\nStart running network... " << std::endl << std::flush;

        Mat img_rgb = Mat(frame.rows, frame.cols, CV_32FC3);
        PRNet::preprocess(frame, img_rgb);

        std::vector<Rect> rects;
        std::vector<Mat> aligned_faces;
        //rects.push_back(Rect(0, 0, frame.cols, frame.rows));

        getFaceBoungingbox(detector, frame, rects);
        if(rects.empty())
            continue;

        {
            SimpleTimer timer("PRNet align total");
            tf_predictor.align(img_rgb, rects, aligned_faces);
        }
        for(auto img:aligned_faces)
            imshow("aligned", img);
        drawBoundingbox(frame, rects);
        imshow("frame", frame);
        waitKey(1);
    }

  return 0;
}
