#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkActor.h>
#include <vector>
#include <QProgressDialog>
#include <QProgressBar>

class ReconstructionPipeline;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoad2DImages();
    void onLoad3DImages();
    void onLoadMultiple2DImages();
    void onRunReconstruction();
    void onShowPointCloud();
    void onClearPointCloud();

private:
    void loadOBJwithMTL(const QString &objPath, const QString &mtlPath);
    void clear3DModel();
    void clear2DTexture();
    void clearPointCloud();

    Ui::MainWindow *ui;
    QVTKOpenGLNativeWidget *vtkWidget;
    vtkSmartPointer<vtkRenderer> renderer;

    std::vector<vtkSmartPointer<vtkActor>> modelActors;
    vtkSmartPointer<vtkActor> texturePlaneActor;
    vtkSmartPointer<vtkActor> cloudActor;

    ReconstructionPipeline *reconstruction;
    bool pointCloudVisible;

    QProgressDialog *progressDialog;
    QProgressBar *progressBar;
    QString lastUsedPath;   // Lưu đường dẫn mở file gần nhất
};

#endif
