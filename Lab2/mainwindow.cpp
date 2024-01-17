#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QDebug>
#include <opencv2/opencv.hpp>
using namespace cv;
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
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
MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_open_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\exp3");
    path = tmp;

    img.load(tmp);
    ui->org->setText("原始图片");
    ui->p1->setPixmap(change_type(img));
    ui->p1->setScaledContents(true);

}

Mat MyConv(const Mat& inputImage, const Mat& kernel)
{
     Mat outputImage = Mat::zeros(inputImage.size(), inputImage.type());

     int kernelSize = kernel.rows;
     int border = kernelSize / 2;

     for (int y = border; y < inputImage.rows - border; y++) {
        for (int x = border; x < inputImage.cols - border; x++) {
            int sum = 0;
            for (int i = -border; i <= border; i++) {
                for (int j = -border; j <= border; j++) {
                    sum += inputImage.at<uchar>(y + i, x + j) * kernel.at<float>(i + border, j + border);//卷积
                }
            }
            outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);
        }
     }

     return outputImage;
}
void MainWindow::on_Sharp_clicked()
{
    QString c_path = change_to_opencv(path);
    Mat input = imread(c_path.toStdString(),0);
    //qDebug()<<input.channels();
    cv::Mat kernel = (Mat_<float>(3, 3) <<
                      1, 1, 1,
                      1, -8, 1,
                      1, 1, 1);
    Mat imglap;
    imglap = MyConv(input, kernel);
   imshow("1",imglap);
    Mat result;
    result = input - imglap;
    //qDebug()<<result.channels();
    QImage img1((uchar *)result.data,result.cols,result.rows,result.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img1);
    ui->Output->setText("对角");
    ui->p2->setPixmap(pix);
    ui->p2->setScaledContents(true);
    kernel = (Mat_<float>(3, 3) <<
                      0, -1, 0,
                      -1, 4, -1,
                      0, -1, 0);

    imglap = MyConv(input, kernel);
   // imshow("1",imglap);
    result = input + imglap;
    //qDebug()<<result.channels();
    QImage img((uchar *)result.data,result.cols,result.rows,result.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix1 =change_type(img);
    ui->No_sharp->setText("output");
    ui->p2_2->setPixmap(pix1);
    ui->p2_2->setScaledContents(true);

}

Mat Unsharp(Mat &input,Mat &highpass,float alpha)
{
    cv::Mat outputImage = cv::Mat::zeros(input.size(), input.type());
    for (int y = 0; y < input.rows; y++)
    {
        for (int x = 0; x < input.cols; x++)
        {
            int newValue = input.at<uchar>(y, x) + cv::saturate_cast<uchar>(alpha * highpass.at<uchar>(y, x)) ;//原图+α*高频部分
            outputImage.at<uchar>(y, x) = cv::saturate_cast<uchar>(newValue);
        }
    }

    return outputImage;
}
void MainWindow::on_UnSharp_clicked()
{
    QString c_path = change_to_opencv(path);
    Mat input = imread(c_path.toStdString(),0);
    cv::Mat kernel = (Mat_<float>(3, 3) <<
                      1, 1, 1,
                      1, 1, 1,
                      1, 1, 1)/9;
    Mat filter ;
   // filter = MyConv(input,kernel);

    blur(input, filter, Size(3,3));
    Mat highpass = input - filter;

    QImage img1((uchar *)filter.data,filter.cols,filter.rows,filter.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix2 =change_type(img1);
    ui->Output->setText("Filter");
    ui->p2->setPixmap(pix2);
    ui->p2->setScaledContents(true);
    float a=ui->parameter->value();
    Mat output = Unsharp(input,highpass,a);
    QImage img((uchar *)output.data,output.cols,output.rows,output.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix1 =change_type(img);
    ui->No_sharp->setText("UnSharp");
    ui->p2_2->setPixmap(pix1);
    ui->p2_2->setScaledContents(true);

}

