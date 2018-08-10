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

# 安装bazal编译工具

scp ubuntu@192.168.1.154:/home/ubuntu/Software/bazel-0.15.2-installer-linux-x86_64.sh ~/Downloads
cd ~/Downloads 
chmod +x bazel-0.15.2-installer-linux-x86_64.sh
./bazel-0.15.2-installer-linux-x86_64.sh


# 克隆tf源码
git clone http://gitlab.bmi/VisionAI/soft/tensorflow.git tensorflow
cd tensorflow & git checkout v1.8.0

# 拷贝cuda的libdevice.10.bc，否则编译安装tensorflow GPU版本时报错：Cannot find libdevice.10.bc under /usr/local/cuda-8.0
sudo cp /usr/local/cuda-8.0/nvvm/libdevice/libdevice.compute_50.10.bc /usr/local/cuda-8.0/libdevice.10.bc
./configure  # 此处会有一系列配置问答，注意cuda部分的选项参数需要手动输入，其他默认回车即可
bazel build --config=cuda --config monolithic tensorflow/libtensorflow_cc.so

# 手动install
# 此处未测试，保守起见可参考pro，只需修改tf源码路径为本机即可
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
# tensorflow
INCLUDEPATH += /home/lirui/packages/tensorflow \
/home/lirui/packages/tensorflow/bazel-tensorflow \
/home/lirui/packages/tensorflow/bazel-bin/tensorflow \
/home/lirui/packages/tensorflow/bazel-genfiles \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/protobuf_archive/src \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/eigen_archive \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/nsync/public \
/home/lirui/packages/tensorflow/bazel-tensorflow
LIBS += -L/home/lirui/packages/tensorflow/bazel-bin/tensorflow

LIBS += -ltensorflow_cc \
        -L/usr/local/lib -lprotobuf

#opencv
INCLUDEPATH += /usr/local/include
LIBS += /usr/local/lib/libopencv_*.so

# mobilenet-SSD-mxnet 人脸检测
INCLUDEPATH += /home/lirui/packages/mxnet-1.1.0/include
LIBS += -L/home/lirui/packages/mxnet-1.1.0/lib -lmxnet

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
//*****must called in your main function
tensorflow::port::InitMain(argv[0], &argc, &argv);
PRNet tf_predictor;
if(0!=tf_predictor.init(pb_model, uv_files, 0))
{
    std::cout << "Initialized tf model fails" << std::endl;
    return -1;
}
Mat img_rgb = Mat(frame.rows, frame.cols, CV_32FC3);
PRNet::preprocess(frame, img_rgb);
vector<Mat> img_batch(1, img_rgb);
vector<Rect> rects;
/* detec face with your detector, and copy to rects*/
vector<vector<Rect> > rects_batch(1, rects);
vector<vector<Mat1f > > kpts_batch;
{
    SimpleTimer timer("PRNet align total");
    tf_predictor.predict(img_batch, rects_batch, kpts_batch);
}
int i=0;
for(auto kpt:kpts_batch[0]){
    Mat roi = frame(rects[i++]);
    DrawKpt(roi, kpt);
}
drawBoundingbox(frame, rects);
imshow("frame", frame);
waitKey(1);
```




## TODO
- [ ] tf->mxnet
- [x] batch
