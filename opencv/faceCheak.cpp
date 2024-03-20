#include <iostream>
#include "opencv2/opencv.hpp"
#include "face.h"
#include <curl/curl.h>
#include <jsoncpp/json/json.h>
#include <unistd.h>
#include <ctime>

using namespace std;
using namespace cv;
using namespace aip;

int main(){

    /*打开摄像头*/
    VideoCapture cap(0);//打开默认摄像头，默认编号为0
    if(!cap.isOpened()){//iSOpened()摄像头打开检测
        cout<<"Camera open failed"<<endl;
        return -1;
    }
    cout << "Camera open success"<<endl;

/*拍摄视频*/
    Mat ColorImage;//mat类型为opencv的图片类
    Mat GrayImage;//gray灰度图片容器

/*利用opencv获取图片并转码*/
    //以下对象用于储存找到的人脸,传参为opencv内置人脸模型文件
    CascadeClassifier Classifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml");    
    vector<Rect> AllFace;//储存标识人脸方框的容器
    vector<uchar> JpgFace;//储存转码为jpg的容器
     Mat MatFace;//截取人脸，将发送到百度人脸识别平台

/*发送到百度人脸接口进行识别*/    
    string app_id = "56496575";
	string api_key ="b03Av4aKnRGWoQXFQjQqUJKy";
	string secres_key ="mHPlkBVdIMV41OYBiaxpkmNnVBNDhlN4";
    aip::Face client(app_id,api_key,secres_key);//通过id链接百度智能云接口

    Json::Value options;
    options["max_face_num"] = "2";
    options["face_type"] = "LIVE";
    options["liveness_control"] = "LOW";
    
    string Base64Face;//转储Base64编码，百度仅识别该编码
    Json::Value result;//json容器用于接收百度返回的消息
    time_t sec;//记录时间s

   
    while(true){

    cap>>ColorImage;//将摄像头拍摄的照片传入照片对象
    cvtColor(ColorImage,GrayImage,CV_BGR2GRAY);//将彩图转为黑白图
    equalizeHist(GrayImage,GrayImage);//对图像进行均衡化处理，调整对比度等以让人脸更明显
    
    Classifier.detectMultiScale(GrayImage,AllFace);//调用人脸检测函数，并分配一个方框
    if(AllFace.size())//如果扫描到人脸则圈出，否则正常播放视频（如果不设置该if语句未检测到人脸则会核心段错误）
    {
        //opencv人像截取
        rectangle(ColorImage,AllFace[0],Scalar(255,255,255));//圈出人脸，参数1灰度图，参数2检测到的脸的方框，参数3方框的颜色
        MatFace = GrayImage(AllFace[0]);//截图人脸数据到MatFace
        imencode(".jpg",MatFace,JpgFace);//将截图的mat容器图片转为jpg格式的图像
        Base64Face = base64_encode((char *)JpgFace.data(),JpgFace.size());//将人脸照片转为base64编码，以便百度平台读取

        //百度人像比对
        result = client.face_search_v3(Base64Face, "BASE64", "scientist",result);//向百度发送人脸识别请求
        // cout<<result<<endl;//打印回发
        if(!result["result"].isNull())//如果百度返回消息非空则为可读取消息
        {
            if(result["result"]["user_list"][0]["score"].asInt() > 80){//如果检测相似程度大于80
                cout<<result["result"]["user_list"][0]["user_id"]<< endl;
                sec = time(NULL);   
                cout<< ctime(&sec)<< endl;
                putText(ColorImage,result["result"]["user_list"][0]["user_id"].asString() , Point(0,50) , FONT_HERSHEY_SIMPLEX , 1.1 , Scalar(0,255,0));
                putText(ColorImage,ctime(&sec) , Point(0,100) , FONT_HERSHEY_SIMPLEX , 1.1 , Scalar(0,255,0));//显示时间
            }
        }

        // cout<< result <<endl;
    }
  
    imshow("faceChack",ColorImage);//实时显示录像

    waitKey(12);//窗口阻塞16.7ms拍摄一帧，当前帧率60
    // sleep(5);
    }///whiel


    
    return 0;
}