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

HEADERS += \
  ./src/tf_predictor.h \
  ./src/stb_image_write.h \
  ./src/stb_image.h \
  ./src/mesh.h \
  ./src/image_impl.h \
  ./src/image.h \
  ./src/face_frontalizer.h \
  ./src/face_cropper.h \
  ./src/face-data.h \
  ./src/cxxopts.hpp

SOURCES += \
 ./src/tf_predictor.cc \
 ./src/main.cc \
 ./src/face_frontalizer.cc \
 ./src/face_cropper.cc \
 ./src/face-data.cc

CONFIG(debug, debug|release) {
    DESTDIR = $$PWD/build/debug
}
else {
    DESTDIR = $$PWD/build/release
}

INCLUDEPATH += ./src

INCLUDEPATH += /home/lirui/packages/tensorflow \
/home/lirui/packages/tensorflow/bazel-tensorflow \
/home/lirui/packages/tensorflow/bazel-bin/tensorflow \
/home/lirui/packages/tensorflow/bazel-genfiles \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/protobuf_archive/src \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/eigen_archive \
/home/lirui/packages/tensorflow/bazel-tensorflow/external/nsync/public \
/home/lirui/packages/tensorflow/bazel-tensorflow

LIBS += -L/home/lirui/packages/tensorflow/bazel-bin/tensorflow -ltensorflow_cc

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib -lprotobuf
LIBS += /usr/local/lib/libopencv_*.so
