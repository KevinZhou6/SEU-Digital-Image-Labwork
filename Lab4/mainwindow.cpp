#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <vector>
#include <QMessageBox>
#include <vector>
#include <QPair>
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

void MainWindow::on_open_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\Morphology");
    path = tmp;

    img.load(tmp);
    ui->p1->setPixmap(change_type(img));
    ui->l1->setText("Input");
    ui->p1->setScaledContents(true);

}

Mat res;
void MainWindow::on_Filter_clicked()
{
    Mat img = imread(change_to_opencv(path).toStdString(),0);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));

    //dilate(src,dst,kernel);

    erode(img, img, kernel);
    QImage img1((uchar *)img.data,img.cols,img.rows,img.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img1);

    ui->l2->setText("Erode");

    ui->p2->setPixmap(pix);
    ui->l2->setScaledContents(true);

    dilate(img,img,kernel);
    QImage img2((uchar *)img.data,img.cols,img.rows,img.cols * 1,QImage::Format_Grayscale8);
     pix =change_type(img2);

    ui->l3->setText("Dilate");

    ui->p3->setPixmap(pix);
    ui->l3->setScaledContents(true);

    dilate(img,img,kernel);
    QImage img3((uchar *)img.data,img.cols,img.rows,img.cols * 1,QImage::Format_Grayscale8);
    pix =change_type(img3);

    ui->l4->setText("Dilate2");

    ui->p4->setPixmap(pix);
    ui->l4->setScaledContents(true);
    erode(img, img, kernel);
    QImage img4((uchar *)img.data,img.cols,img.rows,img.cols * 1,QImage::Format_Grayscale8);
     pix =change_type(img4);

    ui->l5->setText("Erode2");

    ui->p5->setPixmap(pix);
    ui->l5->setScaledContents(true);

    res =img;





}
Mat skeletonize(const cv::Mat &img) {
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);

        done = (cv::countNonZero(img) == 0);
    } while (!done);

    return skel;
}

Mat skeletonReconstruction(const cv::Mat &skeleton, const cv::Mat &element, int iterations) {
    cv::Mat last = cv::Mat::zeros(skeleton.size(), CV_8UC1);
    cv::Mat dilated = skeleton.clone();

    do {
        last = dilated.clone();
        cv::dilate(dilated, dilated, element);
        cv::min(dilated, skeleton, dilated);
    } while (cv::countNonZero(dilated != last) > 0 && --iterations);

    return dilated;
}

void MainWindow::on_pushButton_2_clicked()
{

    Mat origin =res;
    cv::threshold(origin, origin, 127, 255, cv::THRESH_BINARY);

    // 骨架化
    Mat skel = skeletonize(origin);

    // 骨架恢复
    int size = 7;
    Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(size, size));
    Mat restored = skeletonReconstruction(skel, element, 20); // 迭代20次
    Mat output ;
    dilate(restored, output, element);

    QImage img4((uchar *)output.data,output.cols,output.rows,output.cols * 1,QImage::Format_Grayscale8);
    QPixmap pix =change_type(img4);

    ui->l6->setText("Fill");

    ui->p6->setPixmap(pix);
    ui->l6->setScaledContents(true);



}




