#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <QVTKOpenGLNativeWidget.h>
#include <QPushButton>
#include <QProgressBar>
#include <QProgressDialog>
#include <QTimer>
#include <QComboBox>
#include <QList>
#include <QJsonObject>
#include <QDockWidget>
#include <QTextBrowser>
#include <QLineEdit>

class ReconstructionPipeline;
class AIProcessor;
class AIAssistant;
class CrosshairManager;
class CrosshairInteractorStyle;
class PanStyle;
class vtkActor;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum class AIMode {
    None,
    Detection,
    Segmentation
};

class MainWindow : public QMainWindow {
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
    void onTrainModel();
    void onObjectDetection();
    void onSegmentation();
    void onHideAIResults();
    void onAutoNext();
    void onAutoPrev();
    void onAutoTimerTimeout();
    void onLoadDicom();
    
    // Chatbot UI Slots
    void onToggleChatbot();
    void onSendChatMessage();
    void onModelSelected(int index);
    void onNewChat();
    void onChatLinkClicked(const QUrl &url);
    void updateChatUI();
    void onAssistantStatusChanged(const QString &status);
    void onAssistantError(const QString &error);

private:
    Ui::MainWindow *ui;
    QVTKOpenGLNativeWidget *vtkWidget;
    vtkSmartPointer<vtkRenderer> renderer;
    
    vtkSmartPointer<vtkRenderer> axialRenderer;
    vtkSmartPointer<vtkRenderer> sagittalRenderer;
    vtkSmartPointer<vtkRenderer> coronalRenderer;

    CrosshairManager *m_crosshair = nullptr;
    vtkSmartPointer<CrosshairInteractorStyle> m_crosshairStyle;

    ReconstructionPipeline *reconstruction;
    AIProcessor *aiProcessor;
    AIAssistant *aiAssistant;

    vtkSmartPointer<vtkActor> cloudActor;
    std::vector<vtkSmartPointer<vtkActor>> modelActors;
    vtkSmartPointer<vtkActor> texturePlaneActor;

    bool pointCloudVisible;
    QString lastUsedPath;
    QString current2DImagePath;
    QStringList imageFileList;
    int currentImageIndex;
    
    QProgressDialog *progressDialog;
    QProgressBar *progressBar;

    QAction *actShowCloud;
    QAction *actHideCloud;
    QAction *actRunDet;
    QAction *actHideDet;
    QAction *actRunSeg;
    QAction *actHideSeg;

    AIMode currentAIMode;
    
    QTimer *autoTimer;
    bool isAutoNext;
    QPushButton *btnPrev, *btnNext, *btnAutoPrev, *btnAutoNext;

    void loadOBJwithMTL(const QString &objPath, const QString &mtlPath);
    void clear3DModel();
    void clear2DTexture();
    void clearPointCloud();
    void resetToSingleRenderer();
    void setupNavigationUI();
    void updateMenuStates();
    void updateNavigationButtons();
    void loadCurrentIndexImage();

    // Chatbot UI
    void setupChatbotUI();
    QDockWidget* chatbotDock;
    QTextBrowser* chatHistory;
    QLineEdit* chatInput;
    QPushButton* btnSendChat;
    QComboBox* modelSelector;
    QPushButton* btnNewChat;
};

#endif // MAINWINDOW_H
