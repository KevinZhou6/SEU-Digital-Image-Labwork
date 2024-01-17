#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <vector>
#include <QMessageBox>
#include <vector>
#include <fstream>
#include <iostream>
#pragma GCC optimize(2)
#pragma GCC optimize(3,"Ofast","inline")
#define _USE_MATH_DEFINES
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

// 一个函数，用来将小端编码的两个字节转换为一个无符号短整型
unsigned short convert_to_short(char* buffer) {
    unsigned short value = 0;
    value |= (unsigned char)buffer[0]; // 将低字节放在低位
    value |= (unsigned char)buffer[1] << 8; // 将高字节放在高位
    return value;
}
unsigned long convert_to_long(char* buffer) {
    unsigned long value = 0;
    value |= (unsigned char)buffer[0]; // 将低字节放在低位
    value |= (unsigned char)buffer[1] << 8; // 将次低字节放在次低位
    value |= (unsigned char)buffer[2] << 16; // 将次高字节放在次高位
    value |= (unsigned char)buffer[3] << 24; // 将高字节放在高位
    return value;
}
// 一个函数，用来读取raw文件的数据，并存储在一个二维向量中
vector<vector<unsigned short>> read_raw_file(const char* filename) {
    vector<vector<unsigned short>> data; // 用来存储数据的二维向量
    ifstream file(filename, ios::binary); // 以二进制模式打开文件
    if (file.is_open()) { // 如果文件打开成功
        char buffer[4]; // 用来读取两个字节的缓冲区
        long width = 0; // 用来存储文件的宽度
        long height = 0; // 用来存储文件的高度
        if (file.read(buffer, 4)) { // 如果成功读取两个字节
            width = convert_to_long(buffer); // 将两个字节转换为文件的宽度
        }
        if (file.read(buffer, 4)) { // 如果成功读取两个字节
            height = convert_to_long(buffer); // 将两个字节转换为文件的高度
        }
        qDebug()<<(width)<<" "<<(height);
        for (int i = 0; i < height; i++) { // 对于每一行
            vector<unsigned short> row; // 用来存储一行数据的向量
            for (int j = 0; j < width; j++) { // 对于每一列
                if (file.read(buffer, 2)) { // 如果成功读取两个字节
                    unsigned short value = convert_to_short(buffer); // 将两个字节转换为一个无符号短整型
                    row.push_back(value); // 将值添加到一行数据的向量中
                }
            }
            data.push_back(row); // 将一行数据的向量添加到二维向量中
        }
        file.close(); // 关闭文件
    }
    else
    {
         qDebug()<<("failed");
    }
    return data; // 返回数据
}

// 一个函数，用来打印二维向量中的数据
void print_data(vector<vector<unsigned short>> data) {
    qDebug()<<(data.size());
    for (auto row : data) { // 对于每一行
        for (auto value : row) { // 对于每一个值
            cout << value << " "; // 打印值和一个空格
        }
        cout << endl; // 打印一个换行符
    }
}
// 一个函数，用来将二维向量中的数据转换为opencv的Mat对象，并保存为jpg格式的图片
Mat save_data_as_jpg(vector<vector<unsigned short>> data, const char* filename) {
    int height = data.size(); // 获取数据的高度（行数）
    int width = data[0].size(); // 获取数据的宽度（列数）
    cv::Mat image(height, width, CV_8UC1); // 创建一个单通道的8位无符号整型的Mat对象
    for (int i = 0; i < height; i++) { // 对于每一行
        for (int j = 0; j < width; j++) { // 对于每一列
            unsigned short value = data[i][j]; // 获取数据的值
            value = value *255/4096 ; // 将数据的值从0-4095转换为0-255
            image.at<uchar>(i, j) = value; // 将值赋给Mat对象的对应位置
        }
    }
    cv::imwrite(filename, image);  // 将Mat对象保存为jpg格式的图片
    return  image;
}
QPixmap change_type(QImage img)
{
    QPixmap res = QPixmap::fromImage(img);
    return res;
}
Mat input,input2;
void MainWindow::on_open_clicked()
{
    QString path =QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\X_ray");
    // const char* filename = "D:\\QT\\Qt_File\\X_ray\\knee.raw"; // raw文件的文件名
    char* filename = path.toLocal8Bit().data();
    vector<vector<unsigned short>> data = read_raw_file(filename); // 读取raw文件的数据
    const char* jpg_filename = "D:\\QT\\Qt_File\\X_ray\\Input.jpg"; // jpg文件的文件名
    input = save_data_as_jpg(data, jpg_filename);
    QImage img1((uchar *)input.data,input.cols,input.rows,input.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img1);
    ui->p1->setPixmap(pix);
    ui->p1->setScaledContents(true);



}


float table1[256];
static void cal_lookup_table1(void)
{
    for (int i = 0; i < 256; i++)
    {
        table1[i] = (float)(i*i);
    }
}

//计算两个0~255的数的绝对差值的查找表
uchar table2[256][256];
static void cal_lookup_table2(void)
{
    for (int i = 0; i < 256; i++)
    {
        for (int j = i; j < 256; j++)
        {
            table2[i][j] = abs(i - j);
            table2[j][i] = table2[i][j];
        }
    }
}

float  MSE_block(Mat m1, Mat m2)
{
    float sum = 0.0;
    for (int j = 0; j < m1.rows; j++)
    {
        uchar *data1 = m1.ptr<uchar>(j);
        uchar *data2 = m2.ptr<uchar>(j);
        for (int i = 0; i < m1.cols; i++)
        {
            sum += table1[table2[data1[i]][data2[i]]];
        }
    }
    sum = sum / (m1.rows*m2.cols);
    return sum;
}

Mat NL_mean(Mat src, Mat &dst, double h, int halfKernelSize, int halfSearchSize)
{
    Mat boardSrc;
    dst.create(src.rows, src.cols, CV_8UC1);
    int boardSize = halfKernelSize + halfSearchSize;
    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展
    double h2 = h*h;

    int rows = src.rows;
    int cols = src.cols;

    cal_lookup_table1();
    cal_lookup_table2();

    for (int j = boardSize; j < boardSize + rows; j++)
    {
        uchar *dst_p = dst.ptr<uchar>(j - boardSize);
        for (int i = boardSize; i < boardSize + cols; i++)
        {
            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize), Range(i - halfKernelSize, i + halfKernelSize));
            double w = 0;
            double p = 0;
            double sumw = 0;

            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++)   //在搜索框内滑动
            {
                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++)
                {
                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize), Range(i + sc - halfKernelSize, i + sc + halfKernelSize));
                    float d2 = MSE_block(patchA, patchB);

                    w = exp(-d2 / h2);
                    p += boardSrc_p[i + sc] * w;
                    sumw += w;
                }
            }

            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);

        }
    }
    return dst;

}
void integralImgSqDiff(Mat src, Mat &dst, int Ds, int t1, int t2, int m1, int n1)
{
    //计算图像A与图像B的差值图C
    Mat Dist2 = src(Range(Ds, src.rows - Ds), Range(Ds, src.cols - Ds)) - src(Range(Ds + t1, src.rows - Ds + t1), Range(Ds + t2, src.cols - Ds + t2));
    float *Dist2_data;
    for (int i = 0; i < m1; i++)
    {
        Dist2_data = Dist2.ptr<float>(i);
        for (int j = 0; j < n1; j++)
        {
            Dist2_data[j] *= Dist2_data[j];  //计算图像C的平方图D
        }
    }
    integral(Dist2, dst, CV_32F); //计算图像D的积分图
}
double fast_exp(double x){
    double d;
    // 先将尾数的后32位抹零。
    *(reinterpret_cast<int*>(&d) + 0) = 0;
    //再计算指数位，移位，加上偏移量和补偿值
    *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * x + 1072632447);
    return d;
}
Mat fastNLmeans(Mat src, Mat &dst, int ds, int Ds, float h,bool f)
{
    if(f)
    {
        fastNlMeansDenoising(src, dst);
         cv::equalizeHist(dst, dst);
        return dst;
    }

    Mat src_tmp;
    src.convertTo(src_tmp, CV_32F);
    int m = src_tmp.rows;
    int n = src_tmp.cols;
    int boardSize = Ds + ds + 1;
    Mat src_board;
    copyMakeBorder(src_tmp, src_board, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);

    Mat average(m, n, CV_32FC1, 0.0);
    Mat sweight(m, n, CV_32FC1, 0.0);

    float h2 = h*h;
    int d2 = (2 * ds + 1)*(2 * ds + 1);

    int m1 = src_board.rows - 2 * Ds;   //行
    int n1 = src_board.cols - 2 * Ds;   //列
    Mat St(m1, n1, CV_32FC1, 0.0);

    for (int t1 = -Ds; t1 <= Ds; t1++)
    {
        int Dst1 = Ds + t1;
        for (int t2 = -Ds; t2 <= Ds; t2++)
        {
            int Dst2 = Ds + t2;
            integralImgSqDiff(src_board, St, Ds, t1, t2, m1, n1);

            for (int i = 0; i < m; i++)
            {
                float *sweight_p = sweight.ptr<float>(i);
                float *average_p = average.ptr<float>(i);
                float *v_p = src_board.ptr<float>(i + Ds + t1 + ds);
                int i1 = i + ds + 1;   //row
                float *St_p1 = St.ptr<float>(i1 + ds);
                float *St_p2 = St.ptr<float>(i1 - ds - 1);

                for (int j = 0; j < n; j++)
                {

                    int j1 = j + ds + 1;   //col
                    float Dist2 = (St_p1[j1 + ds] + St_p2[j1 - ds - 1]) - (St_p1[j1 - ds - 1] + St_p2[j1 + ds]);

                    Dist2 /= (-d2*h2);
                    float w = fast_exp(Dist2);
                    sweight_p[j] += w;
                    average_p[j] += w * v_p[j + Ds + t2 + ds];
                }
            }

        }
    }

    average = average / sweight;
    average.convertTo(dst, CV_8UC1);
    return dst;
}

cv::Mat gammaCorrection(const cv::Mat& inputImage, float gamma) {
    cv::Mat outputImage = inputImage.clone();
    unsigned char lut[256];

    // 创建查找表
    for (int i = 0; i < 256; ++i) {
        lut[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    // 应用伽马校正
    for (int i = 0; i < outputImage.rows; ++i) {
        for (int j = 0; j < outputImage.cols; ++j) {
            outputImage.at<uchar>(i, j) = lut[inputImage.at<uchar>(i, j)];
        }
    }

    return outputImage;
}

cv::Mat CLAHE1(Mat dst)
{
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0); // 设置对比度限制
    clahe->setTilesGridSize(cv::Size(8, 8)); // 设置网格大小
     clahe->apply(dst, dst);
    return dst;
}

void MainWindow::on_open_2_clicked()
{
    QString path =QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\X_ray");

    char* filename = path.toLocal8Bit().data();
    vector<vector<unsigned short>> data = read_raw_file(filename); // 读取raw文件的数据
    const char* jpg_filename = "D:\\QT\\Qt_File\\X_ray\\Input2.jpg"; // jpg文件的文件名
    input2 = save_data_as_jpg(data, jpg_filename);
    QImage img1((uchar *)input2.data,input2.cols,input2.rows,input2.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img1);
    ui->p1_2->setPixmap(pix);
    ui->p1_2->setScaledContents(true);
}


void MainWindow::on_Sharp_clicked()
{
    Mat result =input;
    fastNLmeans(input, result,9,15,9,true);
    float gamma = ui->Knee->value();
    result =  gammaCorrection(result,gamma) ;

    QImage img1((uchar *)result.data,result.cols,result.rows,result.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img1);
    ui->Output->setText("Knee");
    ui->p2->setPixmap(pix);
    ui->p2->setScaledContents(true);


}




void MainWindow::on_adjust_clicked()
{
    Mat result =input2;
    fastNLmeans(input2, result,9,15,9,true);
    float gamma = ui->Lung->value();
   result =  gammaCorrection(result,gamma) ;




    QImage img2((uchar *)result.data,result.cols,result.rows,result.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix1 =change_type(img2);
    ui->Output_2->setText("Lung");
    ui->p2_2->setPixmap(pix1);
    ui->p2_2->setScaledContents(true);
}

