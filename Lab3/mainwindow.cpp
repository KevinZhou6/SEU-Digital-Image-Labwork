#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <vector>

using namespace cv;



using namespace std;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}
QString path;
QImage img;
QString change_to_opencv(QString path)
{
    QString res;
    for(auto x:path)
    {
        if(x=='\\')
        {
            res +="\\";
        }
        else res+=x;

    }
    return res;
}
QPixmap change_type(QImage img)
{
    QPixmap res = QPixmap::fromImage(img);
    return res;
}
void MainWindow::on_choose_Pic_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000");
    path = tmp;

    img.load(tmp);
    ui->l1->setText("原始图片");
    ui->p1->setPixmap(change_type(img));
    ui->p1->setScaledContents(true);
}
double getMSE(const Mat& I1, const Mat& I2) {
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);
    Scalar s = sum(s1);
    double mse = s.val[0] / (double)(I1.size().width * I1.size().height);
    return mse;
}

Scalar getSSIM(const Mat& I1, const Mat& I2) {
    const double C1 = 6.5025, C2 = 58.5225;
    Mat IxIy, IxxIyy, Ixy;
    multiply(I1, I2, IxIy);
    multiply(I1, I1, IxxIyy);
    multiply(I2, I2, Ixy);

    GaussianBlur(IxIy, IxIy, Size(11, 11), 1.5);
    GaussianBlur(IxxIyy, IxxIyy, Size(11, 11), 1.5);
    GaussianBlur(Ixy, Ixy, Size(11, 11), 1.5);

    Mat num, den;
    multiply(IxIy, 2.0, num);
    addWeighted(IxxIyy, 1.0, Ixy, 1.0, C1, num);
    multiply(Ixy, 2.0, den);
    addWeighted(IxxIyy, 1.0, Ixy, 1.0, C1, den);

    addWeighted(num, 1.0, den, 1.0, C2, num);

    Mat ssim_map;
    divide(num, den, ssim_map);

    Scalar ssim = mean(ssim_map);
    return ssim;
}

void MainWindow::on_JPEG_clicked()
{

    cv::Mat img = cv::imread(path.toStdString(),0);
    if (!img.data)
    {
        printf("invalid image!\n");
        return ;
    }

    vector<int>compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    int k = ui->para->value();
    compression_params.push_back(k);
     cv::imwrite("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file1.jpg", img,compression_params);
    Mat res1= imread("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file1.jpg",0);
    imshow("1",res1);
    //qDebug()<<res1.channels();
    QImage img1((uchar *)res1.data,res1.cols,res1.rows,res1.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix2 =change_type(img1);
    ui->l2->setText("JPEG");
    ui->p2->setPixmap(pix2);
    ui->p2->setScaledContents(true);
    double mse =getMSE(img,res1);
    ui->result->setValue(mse);

}

void calc();

void MainWindow::on_JPEG2000_clicked()
{
    cv::Mat img = cv::imread(path.toStdString(),0);
    if (!img.data)
    {
        printf("invalid image!\n");
        return ;
    }
    vector<int>compression_params;
    vector<uchar>data;
    compression_params.push_back(cv::IMWRITE_JPEG2000_COMPRESSION_X1000);
    int k = ui->para->value();
    compression_params.push_back(k);
   // compression_params.push_back(k);
    imencode(".jp2",img,data,compression_params);
    Mat res = imdecode(data,0);
    imshow("1",res);
    cv::imwrite("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file2.jp2", img,compression_params);
    Mat res1= imread("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file2.jp2",0);
    //imshow("1",res1);
    //qDebug()<<res1.channels();
     QImage img1((uchar *)res1.data,res1.cols,res1.rows,res1.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix2 =change_type(img1);
    ui->l3->setText("JPEG2000");
    ui->p3->setPixmap(pix2);
    ui->p3->setScaledContents(true);
    double mse =getMSE(img,res1);
    ui->jpeg2000->setValue(mse);

}
/*
Mat compressToJPEG2000(const cv::Mat &inputImage) {
    // 检查图片是否为空
    if (inputImage.empty()) {
        std::cerr << "Error: The input image is empty." << std::endl;
        return cv::Mat();
    }

    // JPEG2000压缩参数
    std::vector<int> compression_params_jpeg2000;
    compression_params_jpeg2000.push_back(cv::IMWRITE_JPEG2000_COMPRESSION_X1000);
    compression_params_jpeg2000.push_back(10); // 设置JPEG2000的压缩程度

    // 编码到内存缓冲区
    std::vector<uchar> buffer;
    cv::imencode(".jp2", inputImage, buffer, compression_params_jpeg2000);

    // 从缓冲区创建新的Mat对象
    cv::Mat compressedImage = cv::imdecode(buffer, cv::IMREAD_COLOR);

    return compressedImage;
}
*/
void calc()
{
    Mat org =  cv::imread(path.toStdString(),0);
    Mat jpg = imread("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file1.jpg",0);;
    Mat jpg2000= imread("D:\\QT\\Qt_File\\ImageCompression\\JPEG_JPEG2000\\output\\file2.jp2",0);
    double mse =getMSE(org,jpg);
    qDebug()<<"JPEG MSE";
    qDebug()<<mse;
    qDebug()<<"JPEG2000 MSE";
    double mse2 =getMSE(org,jpg2000);
    qDebug()<<mse2;

   Scalar ssim = getSSIM(org, jpg);
    qDebug()<<"JPEG SSIM";
    qDebug()<<ssim[0];
    qDebug()<<"JPEG2000 SSIM";
     ssim = getSSIM(org, jpg2000);
    qDebug()<<ssim[0];
}
