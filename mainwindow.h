#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkActor.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class ReconstructionPipeline;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadImages();          // chọn nhiều ảnh
    void onRunReconstruction();   // chạy pipeline
    void onShowPointCloud();      // hiển thị point cloud đã tạo
    void onClearPointCloud();     // xoá point cloud khỏi scene

private:
    Ui::MainWindow *ui;
    QVTKOpenGLNativeWidget *vtkWidget;
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkActor> objActor;          // actor cho model OBX gốc
    vtkSmartPointer<vtkActor> cloudActor;        // actor cho point cloud

    ReconstructionPipeline *reconstruction;
    bool pointCloudVisible;
};

#endif // MAINWINDOW_H
