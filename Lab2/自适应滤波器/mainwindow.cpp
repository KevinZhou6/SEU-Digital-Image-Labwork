#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <opencv2/opencv.hpp>
#include <QFileDialog>
#include <QDebug>
#include <QImage>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <algorithm>
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
float MSE_block(Mat m1, Mat m2);
Mat NL_mean(Mat src, Mat &dst, double h, int halfKernelSize, int halfSearchSize);
Mat fastNLmeans(Mat src, Mat &dst, int ds, int Ds, float h);
void integralImgSqDiff(Mat src, Mat &dst, int Ds, int t1, int t2, int m1, int n1);
QPixmap change_type(QImage img)
{
    QPixmap res = QPixmap::fromImage(img);
    return res;
}
void MainWindow::on_open_clicked()
{
    QString tmp=QFileDialog::getOpenFileName(this,tr("打开当前文件夹"),"D:\\QT\\Qt_File\\Noise");

//    QImage image(300, 300, QImage::Format_Grayscale8); // 创建一个100x100的灰度图像
//    image.fill(1); // 将图像填充为灰度值为1的颜色

//    // 将图像保存为文件
//    image.save("gray_image1.png");
    path = tmp;
    QImage img;
    img.load(tmp);
    ui->l1->setText("原始图片");
    ui->p1->setPixmap(change_type(img));
    ui->p1->setScaledContents(true);

}

double generateGaussianNoise(double mu, double sigma)
{
    //定义一个特别小的值
    const double epsilon = std::numeric_limits<double>::min();//返回目标数据类型能表示的最逼近1的正数和1的差的绝对值
    static double z0, z1;
    static bool flag = false;
    flag = !flag;
    //flag为假，构造高斯随机变量
    if (!flag)
        return z1 * sigma + mu;
    double u1, u2;
    //构造随机变量

    do
    {
        u1 = rand()*(1.0 / RAND_MAX);
        u2 = rand()*(1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    //flag为真构造高斯随机变量X
    z0 = sqrt(-2.0*log(u1))*cos(2 * CV_PI * u2);
    z1 = sqrt(-2.0*log(u1))*sin(2 * CV_PI * u2);
    return z1 * sigma + mu;
}
//为图像添加高斯噪声
Mat addGaussianNoise(Mat& srcImage)
{
    Mat resultImage = srcImage.clone();    //深拷贝,克隆
    int channels = resultImage.channels();
    //qDebug()<<channels;    //获取图像的通道
    int nRows = resultImage.rows;    //图像的行数

    int nCols = resultImage.cols*channels;   //图像的总列数
    //判断图像的连续性
    if (resultImage.isContinuous())    //判断矩阵是否连续，若连续，我们相当于只需要遍历一个一维数组
    {
        nCols *= nRows;
        nRows = 1;
    }
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {	//添加高斯噪声
            int val = resultImage.ptr<uchar>(i)[j] + generateGaussianNoise(2, 0.8) * 32;
            if (val < 0)
                val = 0;
            if (val > 255)
                val = 255;
            resultImage.ptr<uchar>(i)[j] = (uchar)val;
        }
    }
    return resultImage;
}

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
// fft变换后进行频谱搬移
void fftshift(cv::Mat &plane0, cv::Mat &plane1)
{
    // 以下的操作是移动图像  (零频移到中心)
    int cx = plane0.cols / 2;
    int cy = plane0.rows / 2;
    cv::Mat part1_r(plane0, cv::Rect(0, 0, cx, cy));  // 元素坐标表示为(cx, cy)
    cv::Mat part2_r(plane0, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_r(plane0, cv::Rect(0, cy, cx, cy));
    cv::Mat part4_r(plane0, cv::Rect(cx, cy, cx, cy));

    cv::Mat temp;
    part1_r.copyTo(temp);  //左上与右下交换位置(实部)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);  //右上与左下交换位置(实部)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    cv::Mat part1_i(plane1, cv::Rect(0, 0, cx, cy));  //元素坐标(cx,cy)
    cv::Mat part2_i(plane1, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3_i(plane1, cv::Rect(0, cy, cx, cy));
    cv::Mat part4_i(plane1, cv::Rect(cx, cy, cx, cy));

    part1_i.copyTo(temp);  //左上与右下交换位置(虚部)
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);  //右上与左下交换位置(虚部)
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);
}
//绘制频谱图
Mat draw(Mat test)
{
    // 将实部虚部分开
    cv::Mat plane[] = { test.clone(), cv::Mat::zeros(test.size() , CV_32FC1) };
    cv::Mat complexIm;
    cv::merge(plane, 2, complexIm);
    cv::dft(complexIm, complexIm); // 进行傅立叶变换，结果保存在自身

    cv::split(complexIm, plane);
    // 为了更清楚的显示 频谱的低频部分将频谱迁移居中显示
    fftshift(plane[0], plane[1]);
    // 计算幅值
    cv::Mat mag;
    cv::magnitude(plane[0], plane[1], mag);
    // 幅值对数化：log（1+m），便于观察频谱信息
    mag += Scalar::all(1);
    cv::log(mag, mag);
    //频谱归一化
    cv::normalize(mag, mag, 0,1, NORM_MINMAX);
    // opencv的方式显示
    //imshow("A",mag);
    // Mat转换为QImage
    return mag;

}
Mat drawHist( cv::Mat &srcImg)
{
    //需要计算图像的哪个通道（RGB空间需要确定计算R空间、G空间或B空间）
    // 此处为单通道
    const int channels[1] = { 0 };
    //直方图的bin数目
    int histSize[] = { 256 };
    //单个维度直方图数值的取值范围
    float inRanges[] = { 0, 256 };
    //确定每个维度的取值范围，即横坐标的总数
    const float *ranges[] = { inRanges };
    //输出的直方图，采用Mat类型
    Mat dstHist;
    cv::calcHist(&srcImg, 1, channels, cv::Mat(), dstHist, 1, histSize, ranges, true, false);
    //imshow("1",dstHist);
    //创建一个白底的图像
    Mat drawImage(Size(256, 256), CV_8UC3, Scalar::all(255));
    //先用cv::minMaxLoc函数计算得到直方图后的像素的最大个数
    double maxValue;
    cv::minMaxLoc(dstHist, 0, &maxValue, 0, 0);
    //遍历直方图得到的数据
    for (int i = 0; i < 256; i++) //从第0个bin到第256个bin
    {
        int value = cvRound(256 * 0.9 *(dstHist.at<float>(i) / maxValue));
        //第i个bin的高度
        cv::rectangle(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 0, 0)); //绘制柱状图
    }
    return drawImage;
}
void saltNoise(cv::Mat img, int n)
{
    int x, y;
    for (int i = 0;i < n / 2;i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 255;
        }
        else if (img.type() == CV_8UC3)
        {
            img.at<cv::Vec3b>(y, x)[0] = 255;
            img.at<cv::Vec3b>(y, x)[1] = 255;
            img.at<cv::Vec3b>(y, x)[2] = 255;
        }
    }
}

//椒噪声
void pepperNoise(cv::Mat img, int n)
{
    int x, y;
    for (int i = 0;i < n / 2;i++)
    {
        x = std::rand() % img.cols;
        y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = 0;
        }
        else if (img.type() == CV_8UC3)
        {
            img.at<cv::Vec3b>(y, x)[0] = 0;
            img.at<cv::Vec3b>(y, x)[1] = 0;
            img.at<cv::Vec3b>(y, x)[2] = 0;
        }
    }
}

void GaussianNoiseSingleChannel(const Mat &src, Mat &dst, double u, double v);
void MainWindow::on_Noise_clicked()
{
    QString now = change_to_opencv(path);
    Mat gray = cv::imread(path.toStdString());

    GaussianNoiseSingleChannel(gray, gray, 1, 100.0);
    cvtColor(gray,gray,cv::COLOR_BGR2GRAY);
   //qDebug()<<gray.channels();
    //imshow("1",gray);
    ui->l2->setText("Noise");
    QImage img((uchar *)gray.data,gray.cols,gray.rows,gray.cols * 1,QImage::Format_Grayscale8);
    ui->p2->setPixmap(change_type(img));
    ui->p2->setScaledContents(true);

    gray.convertTo(gray, CV_32FC1);
    Mat k = drawHist(gray);
    imshow("A",k);



}
uchar adaptiveMedianFilter(cv::Mat &img, int row, int col, int kernelSize, int maxSize);
void MainWindow::on_pushButton_clicked()
{
    QString now = change_to_opencv(path);
    Mat gray = cv::imread(path.toStdString());
    GaussianNoiseSingleChannel(gray, gray, 1, 45.0);
    Mat res = gray;
    cvtColor(res,res,cv::COLOR_BGR2GRAY);

    ui->l2->setText("Noise");
    QImage img((uchar *)res.data,res.cols,res.rows,res.cols * 1,QImage::Format_Grayscale8);
    ui->p2->setPixmap(change_type(img));
    ui->p2->setScaledContents(true);

    // 中值处理
    Mat Image = res;
    cv::medianBlur(Image, Image, 7);

    //均值处理
    Mat avg = res;
    blur(avg, avg, Size(11,11));

   ui->l4->setText("Avg");
   QImage img4((uchar *)avg.data,avg.cols,avg.rows,avg.cols * 1,QImage::Format_Grayscale8);
   ui->p4->setPixmap(change_type(img4));
   ui->p4->setScaledContents(true);

   int minSize = 3;
   int maxSize = 7;

    //自适应中值处理
   //Mat adp = res;
   cv::Mat temp = gray.clone();
   std::vector<cv::Mat> bgr;
   cv::split(gray, bgr );
   cv::copyMakeBorder(bgr[0], bgr[0], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
   cv::copyMakeBorder(bgr[1], bgr[1], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
   cv::copyMakeBorder(bgr[2], bgr[2], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
   for (int j = maxSize / 2;j < bgr[0].rows - maxSize / 2;j++)
   {
        for (int i = maxSize / 2;i < bgr[0].cols - maxSize / 2;i++)
        {
            bgr[0].at<uchar>(j, i) = adaptiveMedianFilter(bgr[0], j, i, minSize, maxSize);
            bgr[1].at<uchar>(j, i) = adaptiveMedianFilter(bgr[1], j, i, minSize, maxSize);
            bgr[2].at<uchar>(j, i) = adaptiveMedianFilter(bgr[2], j, i, minSize, maxSize);
        }
   }
   cv::Mat color_dst;
   cv::merge(bgr,color_dst );
   cvtColor(color_dst,color_dst,cv::COLOR_BGR2GRAY);
  // cv::imshow("sdapt_color_dst", color_dst);
   //qDebug()<<color_dst.channels();
   ui->l5->setText("ADP");
   QImage img5((uchar *)color_dst.data,color_dst.cols,color_dst.rows,color_dst.cols * 1,QImage::Format_Grayscale8);
   ui->p5->setPixmap(change_type(img5));
   ui->p5->setScaledContents(true);


    //Nonlocal means
   // NL_Means nl;
    Mat NL_output;
   //fastNlMeansDenoisingColored(gray,NL_output);
    NL_output= fastNLmeans(res, NL_output,9,15,5.5);
    //qDebug()<<NL_output.channels();
    //cvtColor(NL_output,NL_output,cv::COLOR_BGR2GRAY);


   // imshow("6",NL_output);
    ui->l6->setText("NL");
    QImage img6((uchar *)NL_output.data,NL_output.cols,NL_output.rows,NL_output.cols * 1,QImage::Format_Grayscale8);
    ui->p6->setPixmap(change_type(img6));
    ui->p6->setScaledContents(true);




    // 显示

    //cvtColor(Image,Image,cv::COLOR_BGR2GRAY);
    ui->l3->setText("中值");
    QImage img1((uchar *)Image.data,Image.cols,Image.rows,Image.cols * 1,QImage::Format_Grayscale8);
    ui->p3->setPixmap(change_type(img1));
    ui->p3->setScaledContents(true);



}
void GaussianNoiseSingleChannel(const Mat &src, Mat &dst,double u, double v)
{

    Mat noiseImg(src.size(), CV_64FC1);          //存放噪声图像
    RNG rng((unsigned)time(NULL));               //生成随机数 （均值，高斯）
    rng.fill(noiseImg, RNG::NORMAL, u, v);       //随机高斯填充矩阵
    Mat yccImg;                                  //用来进行对原图像的转换
    cvtColor(src, yccImg, COLOR_BGR2YCrCb);      //色彩空间转换；
    Mat sigImg[3];                               //用来存储单通道图像
    split(yccImg, sigImg);                       //将图像分解到单通道,一幅灰度图像
    sigImg[0].convertTo(sigImg[0], CV_64FC1);    //将uchar转为double
    sigImg[0] = sigImg[0] + noiseImg;            //添加高斯噪声（）
    sigImg[0].convertTo(sigImg[0], CV_8UC1);     //Y通道加高斯噪声后图像,自动截断小于零和大于255的值
    Mat gaussianImg(src.size(), CV_8UC3);        //添加高斯噪声的图像；
    merge(sigImg, 3, gaussianImg);               //和并三个通道
    cvtColor(gaussianImg, dst, COLOR_YCrCb2BGR); //色彩空间转换
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
Mat fastNLmeans(Mat src, Mat &dst, int ds, int Ds, float h)
{
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
// 自适应中值滤波器
uchar adaptiveMedianFilter(cv::Mat &img, int row, int col, int kernelSize, int maxSize)
{
    std::vector<uchar> pixels;
    for (int y = -kernelSize / 2;y <= kernelSize / 2;y++)
    {
        for (int x = -kernelSize / 2;x <= kernelSize / 2;x++)
        {
            pixels.push_back(img.at<uchar>(row + y, col + x));
        }
    }

    sort(pixels.begin(), pixels.end());

    auto min = pixels[0];
    auto max = pixels[kernelSize*kernelSize - 1];
    auto med = pixels[kernelSize*kernelSize / 2];
    auto zxy = img.at<uchar>(row, col);
    if (med > min && med < max)
    {
        // to B
        if (zxy > min && zxy < max)
            return zxy;
        else
            return med;
    }
    else
    {
        kernelSize += 2;
        if (kernelSize <= maxSize)
            return adaptiveMedianFilter(img, row, col, kernelSize, maxSize);// 增大窗口尺寸，继续A过程。
        else
            return med;
    }
}



void MainWindow::on_SP_clicked()
{
    QString now = change_to_opencv(path);
    //qDebug()<<now;
    // return;
    Mat gray = cv::imread(path.toStdString());
   // GaussianNoiseSingleChannel(gray, gray, 1, 45.0);


    // 椒盐
    saltNoise(gray, 200000);
    pepperNoise(gray, 200000);
    Mat res = gray;
    cvtColor(res,res,cv::COLOR_BGR2GRAY);
    // imshow("0",res);
    // qDebug()<<res.channels();
    //  qDebug()<<gray.channels();
    ui->l2->setText("Noise");
    QImage img((uchar *)res.data,res.cols,res.rows,res.cols * 1,QImage::Format_Grayscale8);
    ui->p2->setPixmap(change_type(img));
    ui->p2->setScaledContents(true);

    // 中值处理
    Mat Image = res;
    cv::medianBlur(Image, Image, 7);

    //均值处理
    Mat avg = res;
    blur(avg, avg, Size(11,11));
    // qDebug()<<avg.channels();
    // cvtColor(avg,avg,cv::COLOR_BGR2GRAY);
    //imshow("4",avg);//均值滤波
    ui->l4->setText("Avg");
    QImage img4((uchar *)avg.data,avg.cols,avg.rows,avg.cols * 1,QImage::Format_Grayscale8);
    ui->p4->setPixmap(change_type(img4));
    ui->p4->setScaledContents(true);

    int minSize = 3;
    int maxSize = 7;

    //自适应中值处理
    //Mat adp = res;
    cv::Mat temp = gray.clone();
    std::vector<cv::Mat> bgr;
    cv::split(gray, bgr );
    cv::copyMakeBorder(bgr[0], bgr[0], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    cv::copyMakeBorder(bgr[1], bgr[1], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    cv::copyMakeBorder(bgr[2], bgr[2], maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
    for (int j = maxSize / 2;j < bgr[0].rows - maxSize / 2;j++)
    {
        for (int i = maxSize / 2;i < bgr[0].cols - maxSize / 2;i++)
        {
            bgr[0].at<uchar>(j, i) = adaptiveMedianFilter(bgr[0], j, i, minSize, maxSize);
            bgr[1].at<uchar>(j, i) = adaptiveMedianFilter(bgr[1], j, i, minSize, maxSize);
            bgr[2].at<uchar>(j, i) = adaptiveMedianFilter(bgr[2], j, i, minSize, maxSize);
        }
    }
    cv::Mat color_dst;
    cv::merge(bgr,color_dst );
    cvtColor(color_dst,color_dst,cv::COLOR_BGR2GRAY);
    // cv::imshow("sdapt_color_dst", color_dst);
    //qDebug()<<color_dst.channels();
    ui->l5->setText("ADP");
    QImage img5((uchar *)color_dst.data,color_dst.cols,color_dst.rows,color_dst.cols * 1,QImage::Format_Grayscale8);
    ui->p5->setPixmap(change_type(img5));
    ui->p5->setScaledContents(true);


    //Nonlocal means
    // NL_Means nl;
    Mat NL_output;
    //fastNlMeansDenoisingColored(gray,NL_output);
    NL_output= fastNLmeans(res, NL_output,9,15,9);
    //qDebug()<<NL_output.channels();
    //cvtColor(NL_output,NL_output,cv::COLOR_BGR2GRAY);


    // imshow("6",NL_output);
    ui->l6->setText("NL");
    QImage img6((uchar *)NL_output.data,NL_output.cols,NL_output.rows,NL_output.cols * 1,QImage::Format_Grayscale8);
    ui->p6->setPixmap(change_type(img6));
    ui->p6->setScaledContents(true);




    // 显示

    //cvtColor(Image,Image,cv::COLOR_BGR2GRAY);
    ui->l3->setText("中值");
    QImage img1((uchar *)Image.data,Image.cols,Image.rows,Image.cols * 1,QImage::Format_Grayscale8);
    ui->p3->setPixmap(change_type(img1));
    ui->p3->setScaledContents(true);

}

