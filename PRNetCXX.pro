QT += core
QT -= gui

CONFIG += c++11

TARGET = PRNetCXX
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
#DEFINES += DRAW_IMG

HEADERS += \
    ./src/tf_predictor.h \
    ./src/face-data.h \
    ./src/cxxopts.hpp \
    ./src/simple_timer.h \
    ./src/face_aligner.h \
    src/utils.h

SOURCES += \
    ./src/tf_predictor.cc \
    ./src/main.cc \
    ./src/face-data.cc \
    ./src/face_aligner.cpp

CONFIG(debug, debug|release) {
    DESTDIR = $$PWD/build/debug
}
else {
    DESTDIR = $$PWD/build/release
}

INCLUDEPATH += ./src

# tensorflow and dependencies
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
INCLUDEPATH += $$PWD/third_party/mobileSSD_MX/sample

HEADERS += \
    $$PWD/third_party/mobileSSD_MX/sample/mxnet_ssd_classifier.h \
    $$PWD/third_party/mobileSSD_MX/sample/c_predict_api.h

SOURCES += $$PWD/third_party/mobileSSD_MX/sample/mxnet_ssd_classifier.cpp

INCLUDEPATH += /home/lirui/packages/mxnet-1.1.0/include
LIBS += -L/home/lirui/packages/mxnet-1.1.0/lib -lmxnet

# 其他非源码文件
DISTFILES += \
    README.md
