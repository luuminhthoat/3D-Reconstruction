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
class AIProcessor;
class Image2DLoader;
class Model3DLoader;
class QPushButton;
class QAction;

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
    void onHidePointCloud();
    void onNextImage();
    void onPrevImage();
    
    // AI Phase 4 slots
    void onTrainModel();
    void onObjectDetection();
    void onSegmentation();
    void onHideAIResults();
    void onAutoNext();
    void onAutoPrev();
    void onAutoTimerTimeout();
    void onLoadDicom();

private:
    void loadOBJwithMTL(const QString &objPath, const QString &mtlPath);
    void clear3DModel();
    void clear2DTexture();
    void clearPointCloud();

    Ui::MainWindow *ui;
    QVTKOpenGLNativeWidget *vtkWidget;
    vtkSmartPointer<vtkRenderer> renderer; // 3D/Main
    vtkSmartPointer<vtkRenderer> axialRenderer;
    vtkSmartPointer<vtkRenderer> sagittalRenderer;
    vtkSmartPointer<vtkRenderer> coronalRenderer;
    vtkSmartPointer<vtkRenderer> uiRenderer; // Renderer cho đường kẻ và nhãn

    std::vector<vtkSmartPointer<vtkActor>> modelActors;
    vtkSmartPointer<vtkActor> texturePlaneActor;
    vtkSmartPointer<vtkActor> cloudActor;

    ReconstructionPipeline *reconstruction;
    AIProcessor *aiProcessor;
    bool pointCloudVisible;

    QProgressDialog *progressDialog;
    QProgressBar *progressBar;
    QString lastUsedPath;
    QString current2DImagePath;

    // Menu Actions for toggling
    QAction *actShowCloud;
    QAction *actHideCloud;
    QAction *actRunDet;
    QAction *actHideDet;
    QAction *actRunSeg;
    QAction *actHideSeg;

    // Navigation
    QPushButton *btnPrev;
    QPushButton *btnNext;
    QPushButton *btnAutoPrev;
    QPushButton *btnAutoNext;
    QTimer *autoTimer;
    bool isAutoNext; 
    QStringList imageFileList;
    int currentImageIndex;

    enum class AIMode { None, Detection, Segmentation };
    AIMode currentAIMode;

    void updateMenuStates();
    void updateNavigationButtons();
    void loadCurrentIndexImage();
    void setupNavigationUI();
};

#endif
