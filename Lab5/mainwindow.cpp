#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <vector>
#include <QMessageBox>
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

void MainWindow::on_open_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\Segment");
    path = tmp;

    img.load(tmp);
    ui->p1->setPixmap(change_type(img));
    ui->l1->setText("Input");
    ui->p1->setScaledContents(true);

}
Mat Gtsu(Mat img)//基本全局阈值处理
{
    //平均灰度作为初始阈值
    Scalar m = mean(img);
    int N = m[0];
    //计算阈值
    int p = 0, q = 0;
    int r = 0, t = 0;
    int T = 0, Tp = 256;
    do
    {
        Tp = T;
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                if (img.at<uchar>(i, j) < N)
                {
                    p = p + img.at<uchar>(i, j);
                    r++;
                }
                else
                {
                    q = q + img.at<uchar>(i, j);
                    t++;
                }
            }
        }
        T = (p / r + q / t) / 2;
        r = 0;
        t = 0;
        p = 0;
        q = 0;
    } while (abs(T - Tp) > 1);
    //阈值处理
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<uchar>(i, j) < T)
            {
                img.at<uchar>(i, j) = 0;
            }
            else
            {
                img.at<uchar>(i, j) = 255;
            }
        }
    }
    return img;

}

void threshold_otsu(Mat &mat,Mat&res)
{
//求出图像的最大和最小像素值，确定阈值区间
double min_value_mat, max_value_mat;
Point min_position_mat, max_position_mat;
minMaxLoc(mat, &min_value_mat, &max_value_mat, &min_position_mat, &max_position_mat);

vector <double> var(max_value_mat - min_value_mat + 1);    /* 类间方差容器，不能用数组，因为数组要求用常量定义大小，而mat.rows和mat.cols是变量,除非给数组一个足够大的空间。mat.rows*mat.cols也不能少，否则会报错vector下标越界 */
double thresh_value;   //二值化阈值
int m = 0;    //m必须定义在第一层for函数外面，否则每次都会被初始化为0。
for (thresh_value = min_value_mat; thresh_value < max_value_mat; thresh_value++)
{
    double sum = mat.rows*mat.cols;     //图像像素点总数
    double sum_aim = 0, sum_bg = 0;     //目标和背景像素点总数
    double sum_vaule_aim = 0, sum_vaule_bg = 0;     //目标和背景像素点的总灰度
    for (int i = 0; i < mat.rows; i++)
        for (int j = 0; j < mat.cols; j++)
        {
            int vaule = mat.at<uchar>(i, j);
            if (vaule < thresh_value)     //背景
            {
                sum_bg += 1;
                sum_vaule_bg += vaule;
            }
            else       //目标
            {
                sum_aim += 1;
                sum_vaule_aim += vaule;
            }
        }
    double ratio_aim = sum_aim / sum;   //目标像素点所占比例
    double ratio_bg = sum_bg / sum;     //背景像素点所占比例
    double aver_vaule_aim = sum_vaule_aim / sum_aim;     //目标像素点的平均灰度
    double aver_vaule_bg = sum_vaule_bg / sum_bg;        //背景像素点的平均灰度
    double aver_vaule_mat = ratio_aim * aver_vaule_aim + ratio_bg * aver_vaule_bg;    //图像总平均灰度

    //计算每个阈值下的类间方差并保存到var中
    var[m] = ratio_aim * (aver_vaule_aim - aver_vaule_mat)*(aver_vaule_aim - aver_vaule_mat) +
             ratio_bg * (aver_vaule_bg - aver_vaule_mat)*(aver_vaule_bg - aver_vaule_mat);
    m += 1;
}

//找到最大类间方差以及其对应的阈值
double var_max = 0, var_maxnum = 0;
for (int k = 0; k < max_value_mat - min_value_mat; k++)
    if (var[k] > var_max)
    {
        var_max = var[k];
        var_maxnum = k + min_value_mat;
    }
thresh_value = var_maxnum;
 threshold(mat, res, thresh_value, 255, 3);

}

void MainWindow::on_Global_clicked()
{

 QString tmp = change_to_opencv(path);
 Mat img = imread(tmp.toStdString(),0);

 Mat res = Gtsu(img);
 QImage img1((uchar *)res.data,res.cols,res.rows,res.cols * 1,QImage::Format_Grayscale8);
 QPixmap pix =change_type(img1);

 ui->l2->setText("Global");

 ui->p2->setPixmap(pix);
 ui->l2->setScaledContents(true);

}


void MainWindow::on_Ostu_clicked()
{
 QString tmp = change_to_opencv(path);
 Mat img = imread(tmp.toStdString(),0);

 Mat res = Mat::zeros(img.rows, img.cols, CV_8UC1);
 threshold_otsu(img,res);
 QImage img1((uchar *)res.data,res.cols,res.rows,res.cols * 1,QImage::Format_Grayscale8);
 QPixmap pix =change_type(img1);

 ui->l3->setText("OTSU");

 ui->p3->setPixmap(pix);
 ui->l3->setScaledContents(true);

}

