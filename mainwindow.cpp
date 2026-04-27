#include "mainwindow.h"
#include "PanStyle.h"
#include "ReconstructThread.h"
#include "reconstructionpipeline.h"
#include "ui_mainwindow.h"
#include "aiprocessor.h"
#include <QApplication>
#include <QProcess>
#include <QDebug>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QDateTime>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QMenu>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCellArray.h>
#include <vtkImageData.h>
#include <vtkImageReader2.h>
#include <vtkJPEGReader.h>
#include <vtkLight.h>
#include <vtkOBJImporter.h>
#include <vtkOBJReader.h>
#include <vtkPNGReader.h>
#include <vtkPlaneSource.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkTexture.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>
#include "image2dloader.h"
#include "model3dloader.h"

// ------------------------------------------------------------------
// Hàm trợ giúp
// ------------------------------------------------------------------
void MainWindow::clear3DModel() {
  for (auto &actor : modelActors) {
    renderer->RemoveActor(actor);
  }
  modelActors.clear();
}

void MainWindow::clear2DTexture() {
  if (texturePlaneActor) {
    renderer->RemoveActor(texturePlaneActor);
    texturePlaneActor = nullptr;
  }
}

void MainWindow::clearPointCloud() {
  if (cloudActor) {
    renderer->RemoveActor(cloudActor);
    cloudActor = nullptr;
    pointCloudVisible = false;
  }
}

void MainWindow::loadOBJwithMTL(const QString &objPath, const QString &mtlPath) {
    clear3DModel();
    modelActors = Model3DLoader::load(objPath, mtlPath);
    for (auto &actor : modelActors) {
        renderer->AddActor(actor);
    }
    renderer->ResetCamera();
    vtkWidget->renderWindow()->Render();
}


// ------------------------------------------------------------------
// Constructor & Destructor
// ------------------------------------------------------------------
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), pointCloudVisible(false),
      texturePlaneActor(nullptr), currentImageIndex(-1), currentAIMode(AIMode::None) {
  ui->setupUi(this);

  // Layout setup for central widget
  QWidget *central = new QWidget(this);
  setCentralWidget(central);
  QVBoxLayout *mainLayout = new QVBoxLayout(central);
  
  vtkWidget = new QVTKOpenGLNativeWidget(this);
  mainLayout->addWidget(vtkWidget);

  setupNavigationUI();

  // Sử dụng thư mục chứa file thực thi để tìm config.ini
  QString configPath = QApplication::applicationDirPath() + "/config.ini";
  if (!QFile::exists(configPath)) {
      configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
  }
  
  QSettings settings(configPath, QSettings::IniFormat);
  lastUsedPath = settings.value("Paths/lastUsedPath", "").toString();

  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->SetBackground(0.1, 0.2, 0.4);
  vtkWidget->renderWindow()->AddRenderer(renderer);

  // Thêm đèn chiếu sáng
  vtkSmartPointer<vtkLight> headlight = vtkSmartPointer<vtkLight>::New();
  headlight->SetLightTypeToHeadlight();
  headlight->SetIntensity(1.5);
  renderer->AddLight(headlight);

  vtkSmartPointer<vtkLight> light1 = vtkSmartPointer<vtkLight>::New();
  light1->SetLightTypeToSceneLight();
  light1->SetPosition(2.0, 3.0, 4.0);
  light1->SetIntensity(1.0);
  renderer->AddLight(light1);

  vtkSmartPointer<vtkLight> light2 = vtkSmartPointer<vtkLight>::New();
  light2->SetLightTypeToSceneLight();
  light2->SetPosition(-2.0, 1.0, 3.0);
  light2->SetIntensity(0.8);
  renderer->AddLight(light2);

  progressDialog = new QProgressDialog(this);
  progressBar = new QProgressBar(progressDialog);
  progressBar->setFixedHeight(50);
  progressBar->setMinimumWidth(400);
  progressDialog->setBar(progressBar);
  progressDialog->setCancelButton(nullptr);
  progressDialog->setMinimumDuration(0);
  progressDialog->setWindowModality(Qt::WindowModal);
  progressDialog->setRange(0, 100);
  progressDialog->reset();
  progressDialog->setMinimumSize(500, 200);

  // Toolbar
  QToolBar *toolBar = addToolBar("Reconstruction 3d");
  
  // Phase 1 Group
  QToolButton *btnPhase1 = new QToolButton(this);
  btnPhase1->setText("Phase 1: Viewer");
  btnPhase1->setPopupMode(QToolButton::InstantPopup);
  QMenu *menuPhase1 = new QMenu(btnPhase1);
  menuPhase1->addAction("Load 2D images", this, &MainWindow::onLoad2DImages);
  menuPhase1->addAction("Load 3D images", this, &MainWindow::onLoad3DImages);
  btnPhase1->setMenu(menuPhase1);
  toolBar->addWidget(btnPhase1);

  toolBar->addSeparator();

  // Phase 2 Group
  QToolButton *btnPhase2 = new QToolButton(this);
  btnPhase2->setText("Phase 2: Reconstruction");
  btnPhase2->setPopupMode(QToolButton::InstantPopup);
  QMenu *menuPhase2 = new QMenu(btnPhase2);
  menuPhase2->addAction("Load multiple 2D images", this, &MainWindow::onLoadMultiple2DImages);
  menuPhase2->addAction("Run Reconstruction", this, &MainWindow::onRunReconstruction);
  
  QMenu *menuCloud = menuPhase2->addMenu("Point Cloud");
  actShowCloud = menuCloud->addAction("Show Point Cloud", this, &MainWindow::onShowPointCloud);
  actHideCloud = menuCloud->addAction("Hide Point Cloud", this, &MainWindow::onHidePointCloud);
  actHideCloud->setEnabled(false);

  btnPhase2->setMenu(menuPhase2);
  toolBar->addWidget(btnPhase2);

  toolBar->addSeparator();

  // Phase 4 Group
  QToolButton *btnPhase4 = new QToolButton(this);
  btnPhase4->setText("Phase 4: AI Processing");
  btnPhase4->setPopupMode(QToolButton::InstantPopup);
  QMenu *menuPhase4 = new QMenu(btnPhase4);
  menuPhase4->addAction("Train AI Model", this, &MainWindow::onTrainModel);
  
  QMenu *menuDet = menuPhase4->addMenu("Object Detection");
  actRunDet = menuDet->addAction("Run Detection", this, &MainWindow::onObjectDetection);
  actHideDet = menuDet->addAction("Hide Detection", this, &MainWindow::onHideAIResults);
  actHideDet->setEnabled(false);

  QMenu *menuSeg = menuPhase4->addMenu("Segmentation");
  actRunSeg = menuSeg->addAction("Run Segmentation", this, &MainWindow::onSegmentation);
  actHideSeg = menuSeg->addAction("Hide Segmentation", this, &MainWindow::onHideAIResults);
  actHideSeg->setEnabled(false);

  btnPhase4->setMenu(menuPhase4);
  toolBar->addWidget(btnPhase4);

  reconstruction = new ReconstructionPipeline();
  aiProcessor = new AIProcessor();
  
  QString modelsPath = QFileInfo(__FILE__).absolutePath() + "/models";
  aiProcessor->loadDetectionModel(modelsPath + "/yolo11n.onnx");
  aiProcessor->loadSegmentationModel(modelsPath + "/yolo11n-seg.onnx");

  QString objPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cube.obj";
  QString mtlPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cube.mtl";
  if (QFileInfo::exists(objPath)) {
    loadOBJwithMTL(objPath, mtlPath);
  }

    vtkRenderWindowInteractor *interactor = vtkWidget->renderWindow()->GetInteractor();
    if (!interactor) {
        interactor = vtkRenderWindowInteractor::New();
        vtkWidget->renderWindow()->SetInteractor(interactor);
    }
    
    vtkNew<PanStyle> style;
    interactor->SetInteractorStyle(style);
    interactor->Initialize();

    renderer->ResetCamera();
}

MainWindow::~MainWindow() {
  delete reconstruction;
  delete aiProcessor;
  delete ui;
  // delete progressDialog;
}

// ------------------------------------------------------------------
// Slots
// ------------------------------------------------------------------
void MainWindow::onLoad2DImages() {
  QString startDir = lastUsedPath.isEmpty() ? "" : lastUsedPath;
  QString fileName = QFileDialog::getOpenFileName(
      this, "Select 2D Image", startDir, "Images (*.png *.jpg *.jpeg *.bmp)");
  if (fileName.isEmpty())
    return;
  lastUsedPath = QFileInfo(fileName).absolutePath();
  current2DImagePath = fileName;
  QString configPath = QApplication::applicationDirPath() + "/config.ini";
  if (!QFile::exists(configPath)) configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
  
  QSettings settings(configPath, QSettings::IniFormat);
  settings.setValue("Paths/lastUsedPath", lastUsedPath);

  clear3DModel();
  clearPointCloud();
  clear2DTexture();

  currentAIMode = AIMode::None;
  updateMenuStates();

  texturePlaneActor = Image2DLoader::load(fileName);
  if (texturePlaneActor) {
      renderer->AddActor(texturePlaneActor);
      renderer->ResetCamera();
      vtkWidget->renderWindow()->Render();
      
      // Update image list for navigation
      QFileInfo fileInfo(fileName);
      QDir dir = fileInfo.dir();
      QStringList filters;
      filters << "*.png" << "*.jpg" << "*.jpeg" << "*.bmp";
      imageFileList = dir.entryList(filters, QDir::Files, QDir::Name);
      currentImageIndex = imageFileList.indexOf(fileInfo.fileName());
      
      updateNavigationButtons();
  }
}

void MainWindow::onLoad3DImages() {
  QString startDir = lastUsedPath.isEmpty() ? "" : lastUsedPath;
  QString objFileName = QFileDialog::getOpenFileName(
      this, "Select OBJ file", startDir, "OBJ Files (*.obj)");
  if (objFileName.isEmpty())
    return;

  QFileInfo objInfo(objFileName);
  QString mtlFileName = objInfo.path() + "/" + objInfo.completeBaseName() + ".mtl";

  lastUsedPath = objInfo.absolutePath();
  QString configPath = QApplication::applicationDirPath() + "/config.ini";
  if (!QFile::exists(configPath)) configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
  
  QSettings settings(configPath, QSettings::IniFormat);
  settings.setValue("Paths/lastUsedPath", lastUsedPath);

  clear3DModel();
  clearPointCloud();
  clear2DTexture();
  loadOBJwithMTL(objFileName, mtlFileName);
}

void MainWindow::onLoadMultiple2DImages() {
  QString startDir = lastUsedPath.isEmpty() ? "" : lastUsedPath;
  QStringList files = QFileDialog::getOpenFileNames(
      this, "Select Images", startDir, "Images (*.png *.jpg *.bmp)");
  if (files.isEmpty())
    return;
  lastUsedPath = QFileInfo(files.first()).absolutePath();
  QString configPath = QApplication::applicationDirPath() + "/config.ini";
  if (!QFile::exists(configPath)) configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
  
  QSettings settings(configPath, QSettings::IniFormat);
  settings.setValue("Paths/lastUsedPath", lastUsedPath);

  files.sort();

  std::vector<QString> paths;
  for (const auto &f : files)
    paths.push_back(f);
  reconstruction->setImages(paths);

  QFileInfo firstFile(files.first());
  QString folder = firstFile.absolutePath();
  QString baseName = firstFile.completeBaseName();

  QString prefix = baseName;
  int i = prefix.length() - 1;
  while (i >= 0 && prefix[i].isDigit())
    --i;
  if (i >= 0)
    prefix = prefix.left(i + 1);
  else
    prefix.clear();

  QStringList possibleNames;
  if (!prefix.isEmpty()) {
    possibleNames << prefix + "_par.txt";
    possibleNames << prefix.toLower() + "_par.txt";
  }
  possibleNames << "temple_par.txt" << "templeR_par.txt"
                << "dino_par.txt" << "dinoR_par.txt"
                << "par.txt" << "camera_params.txt";

  bool paramsLoaded = false;
  QString loadedParamsPath;
  for (const auto &name : possibleNames) {
    QString paramsPath = folder + "/" + name;
    if (QFileInfo::exists(paramsPath)) {
      paramsLoaded = reconstruction->loadCameraParams(paramsPath);
      if (paramsLoaded) {
        loadedParamsPath = paramsPath;
        break;
      }
    }
  }

  QString msg = QString("Đã tải %1 ảnh.").arg(paths.size());
  if (paramsLoaded) {
    msg += QString("\n\n✅ Camera params: %1")
               .arg(QFileInfo(loadedParamsPath).fileName());
    msg += "\n→ Sẽ dùng ground-truth projection matrices.";
  } else {
    msg += "\n\n⚠️ Không tìm thấy file params phù hợp trong:\n" + folder;
    msg += "\n→ Cần file có dạng: " +
           (prefix.isEmpty() ? "templeSR_par.txt" : prefix + "_par.txt");
    msg += "\n→ Sẽ dùng estimated pose (kém chính xác hơn).";
  }

  QMessageBox::information(this, "Load Images", msg);
}

void MainWindow::onRunReconstruction() {
  progressDialog->setLabelText("Đang tái tạo 3D...");
  progressDialog->setRange(0, 0);
  progressDialog->setMinimumSize(500, 200);
  progressDialog->show();
  QApplication::processEvents();

  ReconstructThread *thread = new ReconstructThread(reconstruction, this);

  // Dùng QMetaObject::invokeMethod để đảm bảo chạy trên main thread
  connect(
      thread, &QThread::finished, this,
      [thread, this]() {
        progressDialog->hide();
        bool success = thread->isSuccess();
        int pointCount = (int)reconstruction->getPointCloud().size();
        thread->deleteLater();

        if (!success) {
          QMessageBox::warning(
              this, "Lỗi",
              "Reconstruction thất bại!\n"
              "Kiểm tra:\n"
              "  • Đã load ít nhất 2 ảnh chưa?\n"
              "  • File camera params có cùng thư mục ảnh không?");
        } else {
          onShowPointCloud();
          QMessageBox::information(
              this, "Thành công",
              QString("Reconstruction hoàn tất!\nTổng số điểm 3D: %1")
                  .arg(pointCount));
        }
      },
      Qt::QueuedConnection); // ← Thêm Qt::QueuedConnection

  thread->start();
}

void MainWindow::onShowPointCloud() {
  auto pts = reconstruction->getPointCloud();
  auto colors = reconstruction->getPointColors();
  if (pts.empty()) {
    QMessageBox::warning(
        this, "Warning",
        "Chưa có point cloud nào. Hãy chạy Reconstruction trước.");
    return;
  }

  clear3DModel();
  clearPointCloud();

  vtkNew<vtkPoints> vtkPoints;
  vtkNew<vtkCellArray> vertices;
  vtkNew<vtkUnsignedCharArray> vtkColors;
  vtkColors->SetNumberOfComponents(3);
  vtkColors->SetName("Colors");

  for (size_t i = 0; i < pts.size(); ++i) {
    vtkPoints->InsertNextPoint(pts[i].x, pts[i].y, pts[i].z);
    vertices->InsertNextCell(1);
    vertices->InsertCellPoint(i);
    if (i < colors.size()) {
      vtkColors->InsertNextTuple3(colors[i][2], colors[i][1], colors[i][0]);
    } else {
      vtkColors->InsertNextTuple3(255, 255, 255);
    }
  }

  vtkNew<vtkPolyData> polyData;
  polyData->SetPoints(vtkPoints);
  polyData->SetVerts(vertices);
  polyData->GetPointData()->SetScalars(vtkColors);

  vtkNew<vtkVertexGlyphFilter> glyphFilter;
  glyphFilter->SetInputData(polyData);
  glyphFilter->Update();

  vtkNew<vtkPolyDataMapper> cloudMapper;
  cloudMapper->SetInputConnection(glyphFilter->GetOutputPort());

  cloudActor = vtkSmartPointer<vtkActor>::New();
  cloudActor->SetMapper(cloudMapper);
  cloudActor->GetProperty()->SetPointSize(3);

  renderer->AddActor(cloudActor);
  renderer->ResetCamera();
  vtkWidget->renderWindow()->Render();
  pointCloudVisible = true;
  updateMenuStates();
}

void MainWindow::onHidePointCloud() {
    clearPointCloud();
    updateMenuStates();
    vtkWidget->renderWindow()->Render();
}

// ------------------------------------------------------------------
// AI Phase 4 Slots
// ------------------------------------------------------------------
void MainWindow::onTrainModel() {
  QString scriptPath = QFileInfo(__FILE__).absolutePath() + "/train_model.py";
  if (!QFileInfo::exists(scriptPath)) {
    QMessageBox::warning(this, "Error", "Training script not found:\n" + scriptPath);
    return;
  }
  QProcess::startDetached("cmd.exe", QStringList() << "/c" << "start" << "cmd.exe" << "/k" << "python" << scriptPath);
}

void MainWindow::onObjectDetection() {
  if (current2DImagePath.isEmpty()) return;
  if (!aiProcessor->isDetectionModelLoaded()) return;

  cv::Mat inputImage = cv::imread(current2DImagePath.toStdString());
  cv::Mat resultImage = aiProcessor->runObjectDetection(inputImage);
  
  QString tempPath = QApplication::applicationDirPath() + "/temp_ai_result.png";
  cv::imwrite(tempPath.toStdString(), resultImage);
  
  clear2DTexture();
  texturePlaneActor = Image2DLoader::load(tempPath);
  if (texturePlaneActor) {
      renderer->AddActor(texturePlaneActor);
      renderer->ResetCamera();
      vtkWidget->renderWindow()->Render();
  }
  currentAIMode = AIMode::Detection;
  updateMenuStates();
}

void MainWindow::onSegmentation() {
  if (current2DImagePath.isEmpty()) return;
  if (!aiProcessor->isSegmentationModelLoaded()) return;

  cv::Mat inputImage = cv::imread(current2DImagePath.toStdString());
  cv::Mat resultImage = aiProcessor->runSegmentation(inputImage);
  
  QString tempPath = QApplication::applicationDirPath() + "/temp_ai_result.png";
  cv::imwrite(tempPath.toStdString(), resultImage);
  
  clear2DTexture();
  texturePlaneActor = Image2DLoader::load(tempPath);
  if (texturePlaneActor) {
      renderer->AddActor(texturePlaneActor);
      renderer->ResetCamera();
      vtkWidget->renderWindow()->Render();
  }
  currentAIMode = AIMode::Segmentation;
  updateMenuStates();
}

void MainWindow::onHideAIResults() {
    currentAIMode = AIMode::None;
    if (!current2DImagePath.isEmpty()) {
        clear2DTexture();
        texturePlaneActor = Image2DLoader::load(current2DImagePath);
        if (texturePlaneActor) {
            renderer->AddActor(texturePlaneActor);
            vtkWidget->renderWindow()->Render();
        }
    }
    updateMenuStates();
}

void MainWindow::setupNavigationUI() {
    QWidget *navWidget = new QWidget(this);
    QHBoxLayout *navLayout = new QHBoxLayout(navWidget);
    
    // Thu hẹp khoảng cách để vùng view lớn hơn
    navLayout->setContentsMargins(5, 5, 5, 5);
    navLayout->setSpacing(10);
    navWidget->setFixedHeight(40); // Cố định chiều cao thanh điều hướng
    
    btnPrev = new QPushButton("Prev", this);
    btnNext = new QPushButton("Next", this);
    
    btnPrev->setEnabled(false);
    btnNext->setEnabled(false);
    
    navLayout->addStretch(); // Đẩy nút vào giữa
    navLayout->addWidget(btnPrev);
    navLayout->addWidget(btnNext);
    navLayout->addStretch();
    
    qobject_cast<QVBoxLayout*>(centralWidget()->layout())->addWidget(navWidget);
    
    connect(btnPrev, &QPushButton::clicked, this, &MainWindow::onPrevImage);
    connect(btnNext, &QPushButton::clicked, this, &MainWindow::onNextImage);
}

void MainWindow::onNextImage() {
    if (currentImageIndex < imageFileList.size() - 1) {
        currentImageIndex++;
        loadCurrentIndexImage();
    }
}

void MainWindow::onPrevImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        loadCurrentIndexImage();
    }
}

void MainWindow::loadCurrentIndexImage() {
    QString folder = QFileInfo(current2DImagePath).absolutePath();
    QString newPath = folder + "/" + imageFileList[currentImageIndex];
    current2DImagePath = newPath;
    
    clear2DTexture();
    texturePlaneActor = Image2DLoader::load(newPath);
    if (texturePlaneActor) {
        renderer->AddActor(texturePlaneActor);
        
        if (currentAIMode == AIMode::Detection) onObjectDetection();
        else if (currentAIMode == AIMode::Segmentation) onSegmentation();
        
        vtkWidget->renderWindow()->Render();
    }
    updateNavigationButtons();
}

void MainWindow::updateNavigationButtons() {
    btnPrev->setEnabled(currentImageIndex > 0);
    btnNext->setEnabled(currentImageIndex >= 0 && currentImageIndex < imageFileList.size() - 1);
}

void MainWindow::updateMenuStates() {
    actShowCloud->setEnabled(!pointCloudVisible);
    actHideCloud->setEnabled(pointCloudVisible);
    
    actRunDet->setEnabled(currentAIMode != AIMode::Detection);
    actHideDet->setEnabled(currentAIMode == AIMode::Detection);
    
    actRunSeg->setEnabled(currentAIMode != AIMode::Segmentation);
    actHideSeg->setEnabled(currentAIMode == AIMode::Segmentation);
}
