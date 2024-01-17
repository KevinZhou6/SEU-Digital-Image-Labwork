# SEU-Digital-Image-Labwork
东南大学数字图像处理实验。Digital Image Process Labwork

# Introduction

## 平台框架
- `QT - 5.15`
-  `OpenCV -4.5.2`
-   `MinGW`

#### OpenCV-4.5.2 配置

<a href = "MinGW+Opencv-4.5.2">配置详细步骤

QT 项目.pro文件添加Opencv 代码 
```py
INCLUDEPATH += D:/QT/OpenCV-MinGW-Build-OpenCV-4.5.2-x64/include
               D:/QT/OpenCV-MinGW-Build-OpenCV-4.5.2-x64/include/opencv2

LIBS +=  D:/QT/OpenCV-MinGW-Build-OpenCV-4.5.2-x64/x64/mingw/lib/lib*.a

```
