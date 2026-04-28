#include "mainwindow.h"
#include <QDebug>
#include <algorithm>
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
#include "dicomloader.h"
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkCornerAnnotation.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkActor2D.h>
#include <vtkProperty2D.h>
#include <vtkCoordinate.h>
#include <vtkTextProperty.h>
#include <vtkLineSource.h>
#include <vtkLineSource.h>
#include <vtkDICOMImageReader.h>

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
      texturePlaneActor(nullptr), currentImageIndex(-1), currentAIMode(AIMode::None),
      isAutoNext(true) {
  ui->setupUi(this);

  autoTimer = new QTimer(this);
  connect(autoTimer, &QTimer::timeout, this, &MainWindow::onAutoTimerTimeout);

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
  menuPhase1->addAction("Load 2D DICOM", this, &MainWindow::onLoadDicom);
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

// ================================================================
// THAY THẾ TOÀN BỘ HÀM onLoadDicom() trong mainwindow.cpp
//
// Thêm vào phần #include ở đầu mainwindow.cpp:
//   #include <algorithm>          // std::max
//   #include <vtkImageData.h>     // nếu chưa có
// ================================================================

void MainWindow::onLoadDicom() {
    QString fileName = QFileDialog::getOpenFileName(
        this, "Select a DICOM file", lastUsedPath, "DICOM (*.dcm *.dicom)");
    if (fileName.isEmpty()) return;

    QFileInfo fileInfo(fileName);
    QString dirPath = fileInfo.absolutePath();
    lastUsedPath = dirPath;
    qDebug() << "DICOM Load triggered for path:" << dirPath;

    // ============================================================
    // 1. Nạp series - trả về vtkImageData (đã ghép đủ slices)
    // ============================================================
    auto volumeData = DicomLoader::loadSeries(dirPath);
    if (!volumeData || volumeData->GetNumberOfPoints() < 1) {
        QMessageBox::critical(this, "Error",
                              "Không thể đọc DICOM từ:\n" + dirPath +
                                  "\n\nKiểm tra thư mục có chứa file .dcm không.");
        return;
    }

    // ============================================================
    // 2. Tính Window/Level từ range thực tế
    //    MicroDicom báo WL=2373 WW=4746 → dùng làm chuẩn
    // ============================================================
    double range[2];
    volumeData->GetScalarRange(range);
    qDebug() << "Data Range:" << range[0] << "to" << range[1];

    double window, level;
    double span = range[1] - range[0];
    if (span < 500) {
        // 8-bit hoặc dải hẹp
        window = span;
        level  = (range[0] + range[1]) / 2.0;
    } else {
        // 16-bit DICOM: dùng WW/WL chuẩn lấy từ MicroDicom
        window = 4746.0;
        level  = 2373.0;
    }
    qDebug() << "Window:" << window << "Level:" << level;

    // ============================================================
    // 3. Clear và thiết lập lại renderers
    // ============================================================
    clear3DModel();
    clear2DTexture();
    clearPointCloud();
    renderer->RemoveAllViewProps();

    if (!axialRenderer)    axialRenderer    = vtkSmartPointer<vtkRenderer>::New();
    if (!sagittalRenderer) sagittalRenderer = vtkSmartPointer<vtkRenderer>::New();
    if (!coronalRenderer)  coronalRenderer  = vtkSmartPointer<vtkRenderer>::New();

    axialRenderer->RemoveAllViewProps();
    sagittalRenderer->RemoveAllViewProps();
    coronalRenderer->RemoveAllViewProps();

    // Viewport layout 2x2:
    //  [Axial    | Sagittal]
    //  [Coronal  | 3D View ]
    axialRenderer->SetViewport   (0.0, 0.5, 0.5, 1.0);
    sagittalRenderer->SetViewport(0.5, 0.5, 1.0, 1.0);
    coronalRenderer->SetViewport (0.0, 0.0, 0.5, 0.5);
    renderer->SetViewport        (0.5, 0.0, 1.0, 0.5);

    axialRenderer->SetBackground   (0, 0, 0);
    sagittalRenderer->SetBackground(0, 0, 0);
    coronalRenderer->SetBackground (0, 0, 0);
    renderer->SetBackground        (0, 0, 0);

    vtkWidget->renderWindow()->AddRenderer(axialRenderer);
    vtkWidget->renderWindow()->AddRenderer(sagittalRenderer);
    vtkWidget->renderWindow()->AddRenderer(coronalRenderer);

    // ============================================================
    // 4. Nhãn tiêu đề
    // ============================================================
    auto createTitle = [&](vtkRenderer* ren, const QString &text) {
        vtkNew<vtkCornerAnnotation> anno;
        anno->SetText(2, text.toStdString().c_str());
        anno->GetTextProperty()->SetColor(0, 1, 0);
        anno->GetTextProperty()->SetFontSize(18);
        ren->AddViewProp(anno);
    };
    createTitle(axialRenderer,    "Axial");
    createTitle(sagittalRenderer, "Sagittal");
    createTitle(coronalRenderer,  "Coronal");
    createTitle(renderer,         "3D View");

    // ============================================================
    // 5. Đường ngăn cách
    // ============================================================
    auto addBorder = [&](vtkRenderer* ren, bool right, bool bottom) {
        auto makeLine = [&](double x1, double y1, double x2, double y2) {
            vtkNew<vtkLineSource> line;
            line->SetPoint1(x1, y1, 0);
            line->SetPoint2(x2, y2, 0);
            vtkNew<vtkCoordinate> coord;
            coord->SetCoordinateSystemToNormalizedViewport();
            vtkNew<vtkPolyDataMapper2D> m;
            m->SetInputConnection(line->GetOutputPort());
            m->SetTransformCoordinate(coord);
            vtkNew<vtkActor2D> a;
            a->SetMapper(m);
            a->GetProperty()->SetColor(1.0, 1.0, 0.0);
            a->GetProperty()->SetLineWidth(3.0);
            ren->AddActor(a);
        };
        if (right)  makeLine(1.0, 0.0, 1.0, 1.0);
        if (bottom) makeLine(0.0, 0.0, 1.0, 0.0);
    };
    addBorder(axialRenderer,    true,  true);
    addBorder(sagittalRenderer, false, true);
    addBorder(coronalRenderer,  true,  false);

    // ============================================================
    // 6. MPR slices
    // ============================================================
    axialRenderer->AddViewProp(DicomLoader::createSlice(volumeData, 2, window, level));
    sagittalRenderer->AddViewProp(DicomLoader::createSlice(volumeData, 0, window, level));
    coronalRenderer->AddViewProp (DicomLoader::createSlice(volumeData, 1, window, level));

    // ============================================================
    // 7. Volume 3D
    // ============================================================
    renderer->AddViewProp(DicomLoader::createVolume(volumeData, range));

    // ============================================================
    // 8. Camera setup
    // ============================================================
    double center[3];
    volumeData->GetCenter(center);
    double bounds[6];
    volumeData->GetBounds(bounds);
    double maxDim = std::max({bounds[1]-bounds[0],
                              bounds[3]-bounds[2],
                              bounds[5]-bounds[4]});

    auto setupCam = [&](vtkRenderer* ren,
                        double nx, double ny, double nz,
                        double ux, double uy, double uz) {
        vtkCamera* cam = ren->GetActiveCamera();
        cam->SetFocalPoint(center[0], center[1], center[2]);
        double dist = maxDim * 2.0;
        cam->SetPosition(center[0]+nx*dist, center[1]+ny*dist, center[2]+nz*dist);
        cam->SetViewUp(ux, uy, uz);
        cam->ParallelProjectionOn();
        ren->ResetCamera();
    };

    setupCam(axialRenderer,     0,  0,  1,  0, -1,  0);  // Z+ (từ trên xuống)
    setupCam(sagittalRenderer, -1,  0,  0,  0,  0,  1);  // X- (từ trái sang)
    setupCam(coronalRenderer,   0, -1,  0,  0,  0,  1);  // Y- (từ trước ra)

    {
        vtkCamera* cam = renderer->GetActiveCamera();
        cam->SetFocalPoint(center[0], center[1], center[2]);
        cam->SetPosition(center[0]+maxDim*1.5,
                         center[1]-maxDim*1.5,
                         center[2]+maxDim*1.5);
        cam->SetViewUp(0, 0, 1);
        cam->ParallelProjectionOff();
        renderer->ResetCamera();
    }

    // ============================================================
    // 9. Render
    // ============================================================
    vtkWidget->renderWindow()->Render();

    // Lưu lastUsedPath
    QString configPath = QApplication::applicationDirPath() + "/config.ini";
    if (!QFile::exists(configPath))
        configPath = QFileInfo(__FILE__).absolutePath() + "/config.ini";
    QSettings settings(configPath, QSettings::IniFormat);
    settings.setValue("Paths/lastUsedPath", lastUsedPath);

    int* ext = volumeData->GetExtent();
    QMessageBox::information(this, "DICOM Loaded",
                             QString("✅ Loaded volume: %1 × %2 × %3 slices\nRange: %4 – %5\nW: %6 / L: %7")
                                 .arg(ext[1]-ext[0]+1).arg(ext[3]-ext[2]+1).arg(ext[5]-ext[4]+1)
                                 .arg((int)range[0]).arg((int)range[1])
                                 .arg((int)window).arg((int)level));
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
    btnAutoPrev = new QPushButton("AutoPrev", this);
    btnAutoNext = new QPushButton("AutoNext", this);
    btnNext = new QPushButton("Next", this);
    
    btnPrev->setEnabled(false);
    btnAutoPrev->setEnabled(false);
    btnAutoNext->setEnabled(false);
    btnNext->setEnabled(false);
    
    // Styling buttons to indicate state later
    btnAutoPrev->setCheckable(true);
    btnAutoNext->setCheckable(true);

    navLayout->addStretch(); 
    navLayout->addWidget(btnPrev);
    navLayout->addWidget(btnAutoPrev);
    navLayout->addWidget(btnAutoNext);
    navLayout->addWidget(btnNext);
    navLayout->addStretch();
    
    qobject_cast<QVBoxLayout*>(centralWidget()->layout())->addWidget(navWidget);
    
    connect(btnPrev, &QPushButton::clicked, this, &MainWindow::onPrevImage);
    connect(btnNext, &QPushButton::clicked, this, &MainWindow::onNextImage);
    connect(btnAutoPrev, &QPushButton::clicked, this, &MainWindow::onAutoPrev);
    connect(btnAutoNext, &QPushButton::clicked, this, &MainWindow::onAutoNext);
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

void MainWindow::onAutoNext() {
    if (autoTimer->isActive() && isAutoNext) {
        autoTimer->stop();
        btnAutoNext->setChecked(false);
    } else {
        isAutoNext = true;
        autoTimer->start(500); // 500ms per image
        btnAutoNext->setChecked(true);
        btnAutoPrev->setChecked(false);
    }
}

void MainWindow::onAutoPrev() {
    if (autoTimer->isActive() && !isAutoNext) {
        autoTimer->stop();
        btnAutoPrev->setChecked(false);
    } else {
        isAutoNext = false;
        autoTimer->start(500);
        btnAutoPrev->setChecked(true);
        btnAutoNext->setChecked(false);
    }
}

void MainWindow::onAutoTimerTimeout() {
    if (isAutoNext) {
        if (currentImageIndex < imageFileList.size() - 1) {
            onNextImage();
        } else {
            autoTimer->stop();
            btnAutoNext->setChecked(false);
        }
    } else {
        if (currentImageIndex > 0) {
            onPrevImage();
        } else {
            autoTimer->stop();
            btnAutoPrev->setChecked(false);
        }
    }
}

void MainWindow::updateNavigationButtons() {
    bool hasImages = !imageFileList.isEmpty();
    btnPrev->setEnabled(currentImageIndex > 0);
    btnNext->setEnabled(currentImageIndex >= 0 && currentImageIndex < imageFileList.size() - 1);
    btnAutoPrev->setEnabled(hasImages);
    btnAutoNext->setEnabled(hasImages);
    
    if (currentImageIndex <= 0 && !isAutoNext) {
        autoTimer->stop();
        btnAutoPrev->setChecked(false);
    }
    if (currentImageIndex >= imageFileList.size() - 1 && isAutoNext) {
        autoTimer->stop();
        btnAutoNext->setChecked(false);
    }
}

void MainWindow::updateMenuStates() {
    actShowCloud->setEnabled(!pointCloudVisible);
    actHideCloud->setEnabled(pointCloudVisible);
    
    actRunDet->setEnabled(currentAIMode != AIMode::Detection);
    actHideDet->setEnabled(currentAIMode == AIMode::Detection);
    
    actRunSeg->setEnabled(currentAIMode != AIMode::Segmentation);
    actHideSeg->setEnabled(currentAIMode == AIMode::Segmentation);
}
