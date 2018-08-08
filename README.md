# Facial Align With PRNet

## Feature
- runtime 10ms/frame on GTX1080
- C++ api


## 依赖
- tensorflow C++ @1.8
- mxnet(for face detect)



## Build and Run

### install tensorflow
```sh

git clone http://gitlab.bmi/VisionAI/soft/tensorflow.git tensorflow
cd tensorflow
git checkout v1.8.0
# 注意cuda部分的选项参数需要手动输入
./configure
​​bazel build --config=cuda --config monolithic tensorflow/libtensorflow_cc.so

# 手动install
sudo mkdir /usr/local/tensorflow
sudo mkdir /usr/local/tensorflow/include
sudo mkdir /usr/local/tensorflow/lib

sudo cp -r bazel-genfiles/ /usr/local/tensorflow/include
cp -r tensorflow /usr/local/tensorflow/include/
cp -r third_party /usr/local/tensorflow/include/tf/
cp -r bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/tensorflow/lib/

```

### prepare mobileSSD-MX
- [mobileSSD-MX by caoqichun](http://gitlab.bmi/caoqichun/mxnet_mobilenetSSD_face)


### build

- PRNetCXX.pro
```
INCLUDEPATH += /home/lirui/packages/tensorflow \
/home/lirui/packages/tensorflow/bazel-tensorflow \
/home/lirui/packages/tensorflow/bazel-bin/tensorflow \
/home/lirui/packages/tensorflow/bazel-genfiles \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/protobuf_archive/src \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/eigen_archive \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/nsync/public \
/home/lirui/packages/tensorflow/bazel-tensorflow
LIBS += -L/home/lirui/packages/tensorflow/bazel-bin/tensorflow
```

- run and commind line params
```sh
./PRNet --graph ../../data/freeze_model.pb --data ../../data/uv-data --image /media/lirui/Program/Datas/Videos/Face201701052.mp4
```

- use in your cpp
```c++
#include "tf_predictor.h"
#include "simple_timer.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <tensorflow/core/platform/init_main.h>
#include <fstream>
using namespace cv;
using namespace prnet;
using namespace std;
//must called in your main function
tensorflow::port::InitMain(argv[0], &argc, &argv);
PRNet tf_predictor;
if(0!=tf_predictor.init(pb_model, uv_files, 0))
{
    std::cout << "Initialized tf model fails" << std::endl;
    return -1;
}
Mat img_rgb = Mat(frame.rows, frame.cols, CV_32FC3);
PRNet::preprocess(frame, img_rgb);
std::vector<Rect> rects;
std::vector<Mat> aligned_faces;
{
    SimpleTimer timer("PRNet align total");
    tf_predictor.align(img_rgb, rects, aligned_faces);
}
```


## TODO
- [ ] tf->mxnet
- [x] batch
