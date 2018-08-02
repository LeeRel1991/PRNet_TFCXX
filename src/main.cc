
#include "cxxopts.hpp"

#include "tf_predictor.h"
#include "simple_timer.h"

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

int main(int argc, char **argv)
{
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string image_filename, pb_model, uv_files;
    if(parseArgvParams(argc, argv, image_filename, pb_model, uv_files) != 0)
        return -1;

    // Load image
    std::cout << "Loading image \"" << image_filename << "\"" << std::endl;

    Mat frame;
    VideoCapture capture;
    capture.open(image_filename);
    if (!capture.isOpened())
    {
       std::cout << "视频读取失败！" << std::endl;
    }

    // Predict
    PRNet tf_predictor;
    if(0!=tf_predictor.init(pb_model, uv_files))
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

        capture >> frame;

        Mat img_float = Mat(frame.rows, frame.cols, CV_32FC3);
        PRNet::preprocess(frame, img_float);

        std::vector<Rect> rects;
        std::vector<Mat> aligned_faces;
        rects.push_back(Rect(0, 0, frame.cols, frame.rows));

        {
            SimpleTimer timer("PRNet align total");
            tf_predictor.align(img_float, rects, aligned_faces);
        }
    }

  return 0;
}
