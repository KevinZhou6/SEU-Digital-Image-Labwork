QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    NL_means.cpp \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h \
    nl_means.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += D:/QT/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/include
               D:/QT/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/include/opencv2
LIBS +=  D:/QT/OpenCV-MinGW-Build-OpenCV-4.0.1-x64/x64/mingw/lib/lib*.a

