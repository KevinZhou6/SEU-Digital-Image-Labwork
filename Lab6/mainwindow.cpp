#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <vector>
#include <QMessageBox>
#include <vector>
#define _USE_MATH_DEFINES
using namespace cv;
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

void mergeImg(Mat& dst, Mat& src1, Mat& src2)
{
    int rows = src1.rows;
    int cols = src1.cols + 5 + src2.cols;
    CV_Assert(src1.type() == src2.type());
    dst.create(rows, cols, src1.type());
    src1.copyTo(dst(Rect(0, 0, src1.cols, src1.rows)));
    src2.copyTo(dst(Rect(src1.cols + 5, 0, src2.cols, src2.rows)));
}


/*
函数说明：一维高斯卷积，对每行进行高斯卷积
函数输入：	img 		输入原图像
            dst 		一维高斯卷积后的输出图像
*/
void gaussianConvolution(Mat& img, Mat& dst)
{
    int nr = img.rows;
    int nc = img.cols;
    int templates[3] = { 1, 2, 1 };

    // 按行遍历除每行边缘点的所有点
    for (int j = 0; j < nr; j++)
    {
        uchar* data = img.ptr<uchar>(j); 			//提取该行地址
        for (int i = 1; i < nc - 1; i++)
        {
            int sum = 0;
            for (int n = 0; n < 3; n++)
            {
                sum += data[i - 1 + n] * templates[n]; 	//相乘累加
            }
            sum /= 4;
            dst.ptr<uchar>(j)[i] = sum;
        }
    }
}


/*
函数说明：高斯滤波器，利用3*3的高斯模版进行高斯卷积
函数输入：	img 		输入原图像
            dst  		高斯滤波后的输出图像
*/
void gaussianFilter(Mat& img, Mat& dst)
{
    // 对水平方向进行滤波
    Mat dst1 = img.clone();
    gaussianConvolution(img, dst1);
    // 图像矩阵转置
    Mat dst2;
    transpose(dst1, dst2);

    // 对垂直方向进行滤波
    Mat dst3 = dst2.clone();
    gaussianConvolution(dst2, dst3);
    // 再次转置
    transpose(dst3, dst);
}


/*
函数说明：用一阶偏导有限差分计算梯度幅值和方向
函数输入：	img 		输入原图像
            gradXY 		输出的梯度幅值
            theta 		输出的梯度方向
*/
void getGrandient(Mat& img, Mat& gradXY, Mat& theta)
{
    gradXY = Mat::zeros(img.size(), CV_8U);
    theta = Mat::zeros(img.size(), CV_8U);

    for (int j = 1; j < img.rows - 1; j++)
    {
        for (int i = 1; i < img.cols - 1; i++)
        {
            double gradY = double(img.ptr<uchar>(j - 1)[i - 1] + 2 * img.ptr<uchar>(j - 1)[i] + img.ptr<uchar>(j - 1)[i + 1] - img.ptr<uchar>(j + 1)[i - 1] - 2 * img.ptr<uchar>(j + 1)[i] - img.ptr<uchar>(j + 1)[i + 1]);
            double gradX = double(img.ptr<uchar>(j - 1)[i + 1] + 2 * img.ptr<uchar>(j)[i + 1] + img.ptr<uchar>(j + 1)[i + 1] - img.ptr<uchar>(j - 1)[i - 1] - 2 * img.ptr<uchar>(j)[i - 1] - img.ptr<uchar>(j + 1)[i - 1]);

            gradXY.ptr<uchar>(j)[i] = sqrt(gradX * gradX + gradY * gradY); 		//计算梯度
            theta.ptr<uchar>(j)[i] = atan(gradY / gradX); 					//计算梯度方向
        }
    }
}


/*
函数说明：NMS非极大值抑制
函数输入：	gradXY 		输入的梯度幅值
            theta 		输入的梯度方向
            dst 		输出的经局部非极大值抑制后的图像
*/
void nonLocalMaxValue(Mat& gradXY, Mat& theta, Mat& dst)
{
    dst = gradXY.clone();
    for (int j = 1; j < gradXY.rows - 1; j++)
    {
        for (int i = 1; i < gradXY.cols - 1; i++)
        {
            double t = double(theta.ptr<uchar>(j)[i]);
            double g = double(dst.ptr<uchar>(j)[i]);
            if (g == 0.0)
            {
                continue;
            }
            double g0, g1;
            if ((t >= -(3 * M_PI / 8)) && (t < -(M_PI / 8)))
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i - 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i + 1]);
            }
            else if ((t >= -(M_PI / 8)) && (t < M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(j)[i - 1]);
                g1 = double(dst.ptr<uchar>(j)[i + 1]);
            }
            else if ((t >= M_PI / 8) && (t < 3 * M_PI / 8))
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i + 1]);
                g1 = double(dst.ptr<uchar>(j + 1)[i - 1]);
            }
            else
            {
                g0 = double(dst.ptr<uchar>(j - 1)[i]);
                g1 = double(dst.ptr<uchar>(j + 1)[i]);
            }

            if (g <= g0 || g <= g1)
            {
                dst.ptr<uchar>(j)[i] = 0.0;
            }
        }
    }
}


/*
函数说明：弱边缘点补充连接强边缘点
函数输入：img 弱边缘点补充连接强边缘点的输入和输出图像
 */
void doubleThresholdLink(Mat& img)
{
    // 循环找到强边缘点，把其领域内的弱边缘点变为强边缘点
    for (int j = 1; j < img.rows - 2; j++)
    {
        for (int i = 1; i < img.cols - 2; i++)
        {
            // 如果该点是强边缘点
            if (img.ptr<uchar>(j)[i] == 255)
            {
                // 遍历该强边缘点领域
                for (int m = -1; m < 1; m++)
                {
                    for (int n = -1; n < 1; n++)
                    {
                        // 该点为弱边缘点（不是强边缘点，也不是被抑制的0点）
                        if (img.ptr<uchar>(j + m)[i + n] != 0 && img.ptr<uchar>(j + m)[i + n] != 255)
                        {
                            img.ptr<uchar>(j + m)[i + n] = 255; //该弱边缘点补充为强边缘点
                        }
                    }
                }
            }
        }
    }

    for (int j = 0; j < img.rows - 1; j++)
    {
        for (int i = 0; i < img.cols - 1; i++)
        {
            // 如果该点依旧是弱边缘点，及此点是孤立边缘点
            if (img.ptr<uchar>(j)[i] != 255 && img.ptr<uchar>(j)[i] != 255)
            {
                img.ptr<uchar>(j)[i] = 0; //该孤立弱边缘点抑制
            }
        }
    }
}


/*
函数说明：用双阈值算法检测和连接边缘
函数输入：	low 		输入的低阈值
            high 		输入的高阈值
            img 		输入的原图像
            dst 		输出的用双阈值算法检测和连接边缘后的图像
 */
void doubleThreshold(double low, double high, Mat& img, Mat& dst)
{
    dst = img.clone();

    // 区分出弱边缘点和强边缘点
    for (int j = 0; j < img.rows - 1; j++)
    {
        for (int i = 0; i < img.cols - 1; i++)
        {
            double x = double(dst.ptr<uchar>(j)[i]);
            // 像素点为强边缘点，置255
            if (x > high)
            {
                dst.ptr<uchar>(j)[i] = 255;
            }
            // 像素点置0，被抑制掉
            else if (x < low)
            {
                dst.ptr<uchar>(j)[i] = 0;
            }
        }
    }

    // 弱边缘点补充连接强边缘点
    doubleThresholdLink(dst);
}


void MainWindow::on_open_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\Canny");
    path = tmp;

    img.load(tmp);
    ui->p1->setPixmap(change_type(img));
    ui->l1->setText("Input");
    ui->p1->setScaledContents(true);
}


void MainWindow::on_Filter_clicked()
{
    Mat img = imread(change_to_opencv(path).toStdString(),0);
    int low = ui->ll1->value();
    int high =ui->ll2->value();
 Mat gauss_img;
    gaussianFilter(img, gauss_img); //高斯滤波器

    // 用一阶偏导有限差分计算梯度幅值和方向
    Mat gradXY, theta;
    getGrandient(gauss_img, gradXY, theta);

    // 局部非极大值抑制
    Mat local_img;
    nonLocalMaxValue(gradXY, theta, local_img);

    // 用双阈值算法检测和连接边缘
    Mat dst;
    doubleThreshold(low, high, local_img, dst);

    // 图像显示
    Mat outImg;
    mergeImg (outImg,img,dst); //图像拼接


 QImage img1((uchar *)dst.data,dst.cols,dst.rows,dst.cols * 1,QImage::Format_Grayscale8);
 QPixmap pix =change_type(img1);

 ui->l2->setText("MyCanny");

 ui->p2->setPixmap(pix);
 ui->l2->setScaledContents(true);

 Mat dstImage, edge;
 Mat res = imread(change_to_opencv(path).toStdString(),0);

 blur(res, res, Size(3,3));

 Canny(res, edge, low, high,3);

 QImage img2((uchar *)edge.data,edge.cols,edge.rows,edge.cols * 1,QImage::Format_Grayscale8);
 QPixmap pix1 =change_type(img2);

 ui->l3->setText("Canny");

 ui->p3->setPixmap(pix1);
 ui->l3->setScaledContents(true);



}
Mat detectEdgesWithSobel(const cv::Mat &inputImage, int scale, int delta) {
 // 创建一个灰度图像来存储边缘检测的结果
 cv::Mat edges;

 // 如果输入是彩色图像，首先转换为灰度
 cv::Mat grayImage;
 if (inputImage.channels() > 1) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
 } else {
        grayImage = inputImage.clone();
 }

 // 使用高斯模糊去噪声
 cv::GaussianBlur(grayImage, grayImage, cv::Size(3, 3), 0, 0);

 // 创建 grad_x 和 grad_y 矩阵
 cv::Mat grad_x, grad_y;
 cv::Mat abs_grad_x, abs_grad_y;

 // 计算梯度
 cv::Sobel(grayImage, grad_x, CV_16S, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
 cv::Sobel(grayImage, grad_y, CV_16S, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);

 // 转换回 CV_8U
 cv::convertScaleAbs(grad_x, abs_grad_x);
 cv::convertScaleAbs(grad_y, abs_grad_y);

 // 合并梯度（近似）
 cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

 return edges;
}
void MainWindow::on_pushButton_2_clicked()
{
  Mat img = imread(change_to_opencv(path).toStdString(),0);
 int scale = 1;
 int delta = 0;
Mat output = detectEdgesWithSobel(img, scale, delta);

 QImage img2((uchar *)output.data,output.cols,output.rows,output.cols * 1,QImage::Format_Grayscale8);
 QPixmap pix1 =change_type(img2);

 ui->l2->setText("Sobel");

 ui->p2->setPixmap(pix1);
 ui->l2->setScaledContents(true);


}

