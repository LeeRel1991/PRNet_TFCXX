#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cxxopts.hpp"

#include "face_cropper.h"
#include "tf_predictor.h"
#include "face-data.h"
#include "mesh.h"
#include "face_frontalizer.h"

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


static bool LoadImage(const std::string &filename, Image<float> &image) {
  // Load image
  int width, height, channels;
  unsigned char *data = stbi_load(filename.c_str(), &width, &height, &channels,
                                  /* required channels */ 3);
  if (!data) {
    std::cerr << "Failed to load image (" << filename << ")" << std::endl;
    return false;
  }

  // Cast
  image.create(size_t(width), size_t(height), size_t(channels));
  image.foreach ([&](int x, int y, int c, float &v) {
    v = static_cast<float>(data[(y * width + x) * channels + c]) / 255.f;
    // TODO(LTE): Do we really need degamma?
//    v = std::pow(v, 2.2f);
  });

  // Free
  stbi_image_free(data);

  return true;
}


// --------------------------------

// Restore position coordinate.
static void RemapPosition(Image<float> *pos_img, const float scale,
                          const float shift_x, const float shift_y)
{

    size_t n = pos_img->getWidth() * pos_img->getHeight();

    //ofstream fout("org.txt");
    for (size_t i = 0; i < n; i++)
    {
        float x = pos_img->getData()[3 * i + 0];
        float y = pos_img->getData()[3 * i + 1];
        float z = pos_img->getData()[3 * i + 2];

        pos_img->getData()[3 * i + 0] = x * scale + shift_x;
        pos_img->getData()[3 * i + 1] = y * scale + shift_y;
        pos_img->getData()[3 * i + 2] = z * scale; // TODO(LTE): Do we need z offset?

        //    fout << "Org [" << i << "] = " << x << ", " << y << ", " << z << std::endl;
        // fout << "Org [" << i << "] = " << pos_img->getData()[3 * i + 0]  << ", " << pos_img->getData()[3 * i + 1] << " , " << pos_img->getData()[3 * i + 2] << std::endl;
    }
}



static void DrawLandmark(Mat &img, const Mat &pos_img, const FaceData& face_data)
{

    const size_t n_pt = face_data.uv_kpt_indices.size() / 2;

    for (size_t i = 0; i < n_pt; i++)
    {
        const uint32_t x_idx = face_data.uv_kpt_indices[i];
        const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];

        const int x = int(pos_img.at<Vec3f>(y_idx, x_idx)[0]);
        const int y = int(pos_img.at<Vec3f>(y_idx, x_idx)[1]);

        // Draw circle
        circle(img, Point(x, y), 3,  Scalar(255,0,0), -1, 0, 0);
    }
}

#define MatFromImage(image) \
    Mat(image.getHeight(), image.getWidth(), CV_32FC3, (unsigned char*)image.getData())


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

    Mat img = imread(image_filename);

    Image<float> inp_img;
    if (!LoadImage(image_filename, inp_img)) {
        return -1;
    }

    // Predict
    PRNet tf_predictor;
    if(0!=tf_predictor.init(pb_model, uv_files))
    {

        std::cout << "Initialized tf model fails" << std::endl;
        return -1;
    }


    for(int i=0; i<10; ++i)
    {
        std::cout << "\nStart running network... " << std::endl << std::flush;

        Image<float> pos_img;
        auto startT = std::chrono::system_clock::now();

        tf_predictor.predict(inp_img, pos_img);

        auto endT = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> ms = endT - startT;
        std::cout << "Image  Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;
//        std::cout << " pos w " << pos_img.getWidth() << " h " << pos_img.getHeight() << "\n";

//        const float kMaxPos = pos_img.getWidth() * 1.1f;
//        RemapPosition(&pos_img, kMaxPos, 0.0f, 0.0f);

        // opencv Mat as interface
        Mat uv_map;
        Mat img_float = Mat(img.rows, img.cols, CV_32FC3);
        PRNet::preprocess(img, img_float);

        std::vector<Rect> rects;
        std::vector<Mat> aligned_faces;
        rects.push_back(Rect(0, 0, img.cols, img.rows));

        startT = std::chrono::system_clock::now();

        tf_predictor.align(img_float, rects, aligned_faces);

        endT = std::chrono::system_clock::now();
        ms = endT - startT;
        std::cout << "Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;

//        Mat out = img.clone();
//        DrawLandmark(out, uv_map, face_data);
//        tmp_in.convertTo(tmp_in, CV_8UC3);
//        imshow("tmp", tmp_in);
//        imshow("img", out);
//        uv_map.convertTo(uv_map, CV_8UC3);
//        imshow("uv map ", uv_map);
//        waitKey(0);

    }

  return 0;
}
