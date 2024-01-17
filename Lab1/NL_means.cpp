//#include "nl_means.h"
//#include <opencv2/opencv.hpp>
//using namespace cv;

//NL_Means::NL_Means()
//{
//}
//float table1[256];
//static void cal_lookup_table1(void)
//{
//    for (int i = 0; i < 256; i++)
//    {
//        table1[i] = (float)(i*i);
//    }
//}

////计算两个0~255的数的绝对差值的查找表
//uchar table2[256][256];
//static void cal_lookup_table2(void)
//{
//    for (int i = 0; i < 256; i++)
//    {
//        for (int j = i; j < 256; j++)
//        {
//            table2[i][j] = abs(i - j);
//            table2[j][i] = table2[i][j];
//        }
//    }
//}

//float  NL_Means:: MSE_block(Mat m1, Mat m2)
//{
//    float sum = 0.0;
//    for (int j = 0; j < m1.rows; j++)
//    {
//        uchar *data1 = m1.ptr<uchar>(j);
//        uchar *data2 = m2.ptr<uchar>(j);
//        for (int i = 0; i < m1.cols; i++)
//        {
//            sum += table1[table2[data1[i]][data2[i]]];
//        }
//    }
//    sum = sum / (m1.rows*m2.cols);
//    return sum;
//}

//Mat NL_Means::NL_mean(Mat src, Mat &dst, double h, int halfKernelSize, int halfSearchSize)
//{
//    Mat boardSrc;
//    dst.create(src.rows, src.cols, CV_8UC1);
//    int boardSize = halfKernelSize + halfSearchSize;
//    copyMakeBorder(src, boardSrc, boardSize, boardSize, boardSize, boardSize, BORDER_REFLECT);   //边界扩展
//    double h2 = h*h;

//    int rows = src.rows;
//    int cols = src.cols;

//    cal_lookup_table1();
//    cal_lookup_table2();

//    for (int j = boardSize; j < boardSize + rows; j++)
//    {
//        uchar *dst_p = dst.ptr<uchar>(j - boardSize);
//        for (int i = boardSize; i < boardSize + cols; i++)
//        {
//            Mat patchA = boardSrc(Range(j - halfKernelSize, j + halfKernelSize), Range(i - halfKernelSize, i + halfKernelSize));
//            double w = 0;
//            double p = 0;
//            double sumw = 0;

//            for (int sr = -halfSearchSize; sr <= halfSearchSize; sr++)   //在搜索框内滑动
//            {
//                uchar *boardSrc_p = boardSrc.ptr<uchar>(j + sr);
//                for (int sc = -halfSearchSize; sc <= halfSearchSize; sc++)
//                {
//                    Mat patchB = boardSrc(Range(j + sr - halfKernelSize, j + sr + halfKernelSize), Range(i + sc - halfKernelSize, i + sc + halfKernelSize));
//                    float d2 = MSE_block(patchA, patchB);

//                    w = exp(-d2 / h2);
//                    p += boardSrc_p[i + sc] * w;
//                    sumw += w;
//                }
//            }

//            dst_p[i - boardSize] = saturate_cast<uchar>(p / sumw);

//        }
//    }
//    return dst;

//}
