#include <iostream>
#include <thread>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

// 定义互斥锁
std::mutex frameMutex;
std::mutex facesMutex;

void captureThread(VideoCapture& cap, Mat& frame) {
    while (true) {
        Mat localFrame;
        cap >> localFrame; // 从摄像头捕获图像
        // 使用互斥锁保护共享数据
        std::lock_guard<std::mutex> lock(frameMutex);
        frame = localFrame;
    }
}

void detectThread(CascadeClassifier& classifier, Mat& frame, vector<Rect>& faces) {
    Mat grayFrame;
    while (true) {
        Mat localFrame;
        {
            // 使用互斥锁保护共享数据
            std::lock_guard<std::mutex> lock(frameMutex);
            localFrame = frame.clone(); // 复制图像以避免数据竞争
        }
        cvtColor(localFrame, grayFrame, COLOR_BGR2GRAY); // 转换为灰度图像
        equalizeHist(grayFrame, grayFrame); // 直方图均衡化
        classifier.detectMultiScale(grayFrame, faces); // 人脸检测
    }
}

void displayThread(VideoCapture& cap, CascadeClassifier& classifier) {
    Mat frame; // 存储捕获的图像
    vector<Rect> faces; // 存储检测到的人脸位置信息
    while (true) {
        {
            // 使用互斥锁保护共享数据
            std::lock_guard<std::mutex> lock(frameMutex);
            frame.copyTo(frame); // 复制图像以避免数据竞争
        }
        {
            // 使用互斥锁保护共享数据
            std::lock_guard<std::mutex> lock(facesMutex);
            faces.swap(faces); // 复制人脸位置信息以避免数据竞争
        }
        for (size_t i = 0; i < faces.size(); i++) {
            rectangle(frame, faces[i], Scalar(255, 255, 255)); // 在图像上绘制人脸框
        }
        imshow("Face Detection", frame); // 显示图像
        waitKey(1); // 等待用户按键
    }
}

int main() {
    VideoCapture cap(0); // 打开默认摄像头，默认编号为0
    if (!cap.isOpened()) {
        cout << "Camera open failed" << endl;
        return -1;
    }
    cout << "Camera open success" << endl;

    Mat frame; // 存储捕获的图像
    CascadeClassifier classifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml"); // 加载人脸检测器模型

    // 创建三个线程
    thread capture(captureThread, ref(cap), ref(frame)); // 图像捕获线程
    thread detect(detectThread, ref(classifier), ref(frame), ref(faces)); // 人脸检测线程
    thread display(displayThread, ref(cap), ref(classifier)); // 图像显示线程

    // 等待线程结束
    capture.join();
    detect.join();
    display.join();

    return 0;
}
