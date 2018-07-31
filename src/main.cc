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

#include <opencv2/opencv.hpp>
#include <fstream>
using namespace cv;
using namespace prnet;
using namespace std;

template <typename T>
inline T clamp(T f, T fmin, T fmax) {
  return std::max(std::min(fmax, f), fmin);
}

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

static bool SaveImage(const std::string &filename, Image<float> &image, const float scale = 1.0f) {
  const size_t height = image.getHeight();
  const size_t width = image.getWidth();
  const size_t channels = image.getChannels();

  // Cast
  std::vector<unsigned char> data(height * width * channels);
  image.foreach ([&](int x, int y, int c, float &v) {
    data[(size_t(y) * width + size_t(x)) * size_t(channels) + size_t(c)] =
        static_cast<unsigned char>(clamp(scale * v * 255.f, 0.0f, 255.0f));
  });

  // Save
  stbi_write_jpg(filename.c_str(), int(width), int(height), int(channels),
                 &data.at(0), 0);

  return true;
}

// --------------------------------

// Create texture map from 3D position map
static bool CreateTexture(const Image<float> &image, const Image<float> &posmap,
                          Image<float> *texture) {
  if (image.getWidth() != 256) {
    std::cerr << "Invalid width for Image. width must be 256 but has "
              << image.getWidth() << std::endl;
    return false;
  }

  if (image.getHeight() != 256) {
    std::cerr << "Invalid height for Image. height must be 256 but has "
              << image.getHeight() << std::endl;
    return false;
  }

  if (image.getChannels() != 3) {
    std::cerr << "Invalid channels for Image. channels must be 3 but has "
              << image.getChannels() << std::endl;
    return false;
  }

  if (posmap.getWidth() != 256) {
    std::cerr << "Invalid width for Position map. width must be 256 but has "
              << posmap.getWidth() << std::endl;
    return false;
  }

  if (posmap.getHeight() != 256) {
    std::cerr << "Invalid height for Position map. height must be 256 but has "
              << posmap.getHeight() << std::endl;
    return false;
  }

  if (posmap.getChannels() != 3) {
    std::cerr
        << "Invalid channels for Position map. channels must be 3 but has "
        << posmap.getChannels() << std::endl;
    return false;
  }

  size_t width = image.getWidth();
  size_t height = image.getHeight();

  texture->create(width, height, /* RGB */3);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      // Look up 2D position from 3D position map.
      float vx = posmap.getData()[3 * (y * width + x) + 0];
      float vy = posmap.getData()[3 * (y * width + x) + 1];
      // float b = image.getData()[3 * (y * width + x) + 2];

      // Fetch corresponding texel color
      // No filtering(nearest neighbor point sampling)
      int px = int(vx);
      int py = int(vy);

      if ((px < 0) || (py < 0) || (px >= int(width)) || (py >= int(height))) {
        // out-of-bounds
        texture->getData()[3 * (y * width + x) + 0] = 0.0f;
        texture->getData()[3 * (y * width + x) + 1] = 0.0f;
        texture->getData()[3 * (y * width + x) + 2] = 0.0f;
      } else {
        texture->getData()[3 * (y * width + x) + 0] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 0];
        texture->getData()[3 * (y * width + x) + 1] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 1];
        texture->getData()[3 * (y * width + x) + 2] =
            image.getData()[3 * (size_t(py) * width + size_t(px)) + 2];
      }
    }
  }

  return true;
}

// Convert 3D position map(Image) to mesh using FaceData.
static bool ConvertToMesh(const Image<float> &image, const FaceData &face_data,
                          Mesh *mesh) {
  if (image.getWidth() != 256) {
    std::cerr << "Invalid width for Image. width must be 256 but has "
              << image.getWidth() << std::endl;
    return false;
  }

  if (image.getHeight() != 256) {
    std::cerr << "Invalid height for Image. height must be 256 but has "
              << image.getHeight() << std::endl;
    return false;
  }

  if (image.getChannels() != 3) {
    std::cerr << "Invalid channels for Image. channels must be 3 but has "
              << image.getChannels() << std::endl;
    return false;
  }

  //// DBG
  // for (size_t i = 0; i < 10; i++) {
  //  float x = image.getData()[3 * i + 0];
  //  float y = image.getData()[3 * i + 1];
  //  float z = image.getData()[3 * i + 2];
  //
  //  std::cout << "[" << i << "] + " << x << ", " << y << ", " << z <<
  //  std::endl;
  //}

  float bmin[3];
  float bmax[3];

  // Look up vertex position from 3D position map(256x256x3)
  mesh->vertices.clear();
  mesh->uvs.clear();
  for (size_t i = 0; i < face_data.face_indices.size(); i++) {
    size_t idx = face_data.face_indices[i];

    uint32_t px = uint32_t(idx % image.getWidth());
    uint32_t py = uint32_t(idx / image.getHeight());

    float x = image.getData()[3 * (py * image.getWidth() + px) + 0];
    float y = image.getData()[3 * (py * image.getWidth() + px) + 1];
    float z = image.getData()[3 * (py * image.getWidth() + px) + 2];

    mesh->vertices.push_back(x);
    mesh->vertices.push_back(y);
    mesh->vertices.push_back(z);

    if (i == 0) {
      bmin[0] = bmax[0] = x;
      bmin[1] = bmax[1] = y;
      bmin[2] = bmax[2] = z;
    } else {
      bmin[0] = std::min(bmin[0], x);
      bmin[1] = std::min(bmin[1], y);
      bmin[2] = std::min(bmin[2], z);

      bmax[0] = std::max(bmax[0], x);
      bmax[1] = std::max(bmax[1], y);
      bmax[2] = std::max(bmax[2], z);
    }

    //normalize uv
    float u = x / float(image.getWidth());
    float v = y / float(image.getHeight());

    mesh->uvs.push_back(u);
    mesh->uvs.push_back(v);
  }

  std::cout << "bmin " << bmin[0] << ", " << bmin[1] << ", " << bmin[2]
            << std::endl;
  std::cout << "bmax " << bmax[0] << ", " << bmax[1] << ", " << bmax[2]
            << std::endl;

#if 1
  // Centerize vertex position.
  {
    float bsize[3];
    bsize[0] = bmax[0] - bmin[0];
    bsize[1] = bmax[1] - bmin[1];
    bsize[2] = bmax[2] - bmin[2];
    for (size_t i = 0; i < mesh->vertices.size() / 3; i++) {
      mesh->vertices[3 * i + 0] -= (bmin[0] + 0.5f * bsize[0]);
      mesh->vertices[3 * i + 1] -= (bmin[1] + 0.5f * bsize[1]);
      mesh->vertices[3 * i + 2] -= (bmin[2] + 0.5f * bsize[2]);
    }
  }
#endif

  for (size_t i = 0; i < face_data.triangles.size(); i++) {
    // It looks triangle index starts with 1, but accepts it.
    uint32_t idx = face_data.triangles[i];
    if (idx >= (mesh->vertices.size() / 3)) {
      std::cerr << "??? : invalid triangle index. " << idx
                << " is greater or equal to " << (mesh->vertices.size() / 3)
                << std::endl;
      exit(-1);
    }
    mesh->faces.push_back(idx);
  }

  return true;
}

// Save as wavefront .obj mesh
static bool SaveAsWObj(const std::string &filename, prnet::Mesh &mesh) {
  std::ofstream ofs(filename);
  if (!ofs) {
    std::cerr << "Failed to open file to write : " << filename << std::endl;
    return false;
  }

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    ofs << "v " << 255.0f * mesh.vertices[3 * i + 0] << " "
        << 255.0f * mesh.vertices[3 * i + 1] << " "
        << 255.0f * mesh.vertices[3 * i + 2] << std::endl;
  }

  for (size_t i = 0; i < mesh.uvs.size() / 2; i++) {
    ofs << "vt " << mesh.uvs[2 * i + 0] << " " << mesh.uvs[2 * i + 1]
        << std::endl;
  }

  for (size_t i = 0; i < mesh.faces.size() / 3; i++) {
    // For .obj, face index starts with 1, so add +1.
    uint32_t f0 = mesh.faces[3 * i + 0] + 1;
    uint32_t f1 = mesh.faces[3 * i + 1] + 1;
    uint32_t f2 = mesh.faces[3 * i + 2] + 1;

    // Assume # of v == # of vt.
    ofs << "f " << f0 << "/" << f0 << " " << f1 << "/" << f1 << " " << f2 << "/"
        << f2 << std::endl;
  }

  // TODO(LTE): Output .mtl file.

  return true;
}

// Restore position coordinate.
static void RemapPosition(Image<float> *pos_img, const float scale,
                          const float shift_x, const float shift_y) {

  size_t n = pos_img->getWidth() * pos_img->getHeight();


  ofstream fout("org.txt");
  for (size_t i = 0; i < n; i++) {
    float x = pos_img->getData()[3 * i + 0];
    float y = pos_img->getData()[3 * i + 1];
    float z = pos_img->getData()[3 * i + 2];

    pos_img->getData()[3 * i + 0] = x * scale + shift_x;
    pos_img->getData()[3 * i + 1] = y * scale + shift_y;
    pos_img->getData()[3 * i + 2] = z * scale; // TODO(LTE): Do we need z offset?


//    fout << "Org [" << i << "] = " << x << ", " << y << ", " << z << std::endl;
    fout << "Org [" << i << "] = " << pos_img->getData()[3 * i + 0]  << ", " << pos_img->getData()[3 * i + 1] << " , " << pos_img->getData()[3 * i + 2] << std::endl;
  }
}

static void DrawLandmark(const Image<float> &cropped_img,
                         const Image<float> &pos_img,
                         const FaceData& face_data,
                         Image<float> *out_img, float radius = 1.f) {
  *out_img = cropped_img;  // copy
  const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
  const int ksize = int(std::ceil(radius));
  for (size_t i = 0; i < n_pt; i++) {
    const uint32_t x_idx = face_data.uv_kpt_indices[i];
    const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];
    const int x = int(pos_img.fetch(x_idx, y_idx, 0));
    const int y = int(pos_img.fetch(x_idx, y_idx, 1));
    // Draw circle
    for (int rx = -ksize; rx <= ksize; rx++) {
      for (int ry = -ksize; ry <= ksize; ry++) {
        if (radius < float(rx * rx + ry * ry)) {
            continue;
        }
        
        out_img->fetch(size_t(x + rx), size_t(y + ry), 0) = 0;
        out_img->fetch(size_t(x + rx), size_t(y + ry), 1) = 255;
        out_img->fetch(size_t(x + rx), size_t(y + ry), 2) = 0;
      }
    }
  }
}



static void DrawLandmark(const Mat &cropped_img,
                         const Mat &pos_img,
                         const FaceData& face_data,
                         Mat &out_img, float radius = 1.f) {
  out_img = cropped_img.clone();  // copy

  const size_t n_pt = face_data.uv_kpt_indices.size() / 2;
  const int ksize = int(std::ceil(radius));
  for (size_t i = 0; i < n_pt; i++) {
    const uint32_t x_idx = face_data.uv_kpt_indices[i];
    const uint32_t y_idx = face_data.uv_kpt_indices[i + n_pt];

    const int x = int(pos_img.at<Vec3f>(y_idx, x_idx)[0]);
    const int y = int(pos_img.at<Vec3f>(y_idx, x_idx)[1]);

    // Draw circle
    circle(out_img, Point(x, y), 3,  Scalar(255,0,0), -1, 0, 0);


  }
}

#define MatFromImage(image) \
    Mat(image.getHeight(), image.getWidth(), CV_32FC3, (unsigned char*)image.getData())

int main(int argc, char **argv)
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

  std::string image_filename = result["image"].as<std::string>();
  std::string graph_filename = result["graph"].as<std::string>();
  std::string data_dirname = result["data"].as<std::string>();

  // Load image
  std::cout << "Loading image \"" << image_filename << "\"" << std::endl;

  Mat img = imread(image_filename);

  Image<float> inp_img;
  if (!LoadImage(image_filename, inp_img)) {
    return -1;
  }

  // Meshing
  FaceData face_data;
  if (!LoadFaceData(data_dirname, &face_data)) {
    return -1;
  }



  // Predict
  TensorflowPredictor tf_predictor;
  tf_predictor.init(argc, argv);
  std::cout << "Initialized" << std::endl;
  tf_predictor.load(graph_filename, "Placeholder", "resfcn256/Conv2d_transpose_16/Sigmoid");
  std::cout << "Loaded model" << std::endl;

  for(int i=0; i<10; ++i)
  {
      std::cout << "Start running network... " << std::endl << std::flush;

      Image<float> pos_img;

      auto startT = std::chrono::system_clock::now();

      tf_predictor.predict(inp_img, pos_img);

      auto endT = std::chrono::system_clock::now();
      std::chrono::duration<double, std::milli> ms = endT - startT;
      std::cout << "Image  Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;
      std::cout << " pos w " << pos_img.getWidth() << " h " << pos_img.getHeight() << "\n";

      const float kMaxPos = pos_img.getWidth() * 1.1f;
      RemapPosition(&pos_img, kMaxPos, 0.0f, 0.0f);

      // opencv Mat as interface

      cv::Mat tmp_in = MatFromImage(inp_img);

      Mat uv_map;
      Mat img_float;
      cvtColor(img, img, COLOR_BGR2RGB);
      img.convertTo(img_float, CV_32FC3);

      for(int i = 0; i < img_float.rows; ++i)
      {
          for(int j= 0; j< img_float.cols; ++j)
          {
             img_float.at<cv::Vec3f>(i, j)[0]=  img_float.at<cv::Vec3f>(i, j)[0]/255.f;
             img_float.at<cv::Vec3f>(i, j)[1]=  img_float.at<cv::Vec3f>(i, j)[1]/255.f;
             img_float.at<cv::Vec3f>(i, j)[2]=  img_float.at<cv::Vec3f>(i, j)[2]/255.f;
             float x= img_float.at<cv::Vec3f>(i, j)[0];
             float y= img_float.at<cv::Vec3f>(i, j)[1];
             float z= img_float.at<cv::Vec3f>(i, j)[2];
          }
      }
      startT = std::chrono::system_clock::now();

      tf_predictor.predict(img_float, uv_map);

      endT = std::chrono::system_clock::now();
      ms = endT - startT;
      std::cout << "Ran network. elapsed = " << ms.count() << " [ms] " << std::endl;

      Mat out_img;
      DrawLandmark(img, uv_map, face_data, out_img );
      tmp_in.convertTo(tmp_in, CV_8UC3);
      imshow("tmp", tmp_in);
      imshow("img", img);
      uv_map.convertTo(uv_map, CV_8UC3);
      imshow("landmarks", out_img);
      imshow("uv map ", uv_map);
      waitKey(0);

    // Draw landmarks
    Image<float> dbg_lmk_image;
    Image<float> color_img = inp_img;
    DrawLandmark(color_img, pos_img, face_data, &dbg_lmk_image);
    SaveImage("landmarks1.jpg", dbg_lmk_image);

  }

  return 0;
}
