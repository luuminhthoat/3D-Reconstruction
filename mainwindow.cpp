#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "reconstructionpipeline.h"
#include <QVBoxLayout>
#include <QToolBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QFileInfo>
#include <QProgressBar>
#include <QStatusBar>
#include <vtkOBJImporter.h>
#include <vtkOBJReader.h>
#include <vtkActorCollection.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkUnsignedCharArray.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkLight.h>
#include <vtkTexture.h>
#include <vtkJPEGReader.h>
#include <vtkPNGReader.h>
#include <vtkImageReader2.h>
#include <vtkPlaneSource.h>
#include <vtkImageData.h>
#include "PanStyle.h"
#include <QTimer>
#include <QThread>

// ------------------------------------------------------------------
// Hàm trợ giúp
// ------------------------------------------------------------------
void MainWindow::clear3DModel()
{
    for (auto &actor : modelActors) {
        renderer->RemoveActor(actor);
    }
    modelActors.clear();
}

void MainWindow::clear2DTexture()
{
    if (texturePlaneActor) {
        renderer->RemoveActor(texturePlaneActor);
        texturePlaneActor = nullptr;
    }
}

void MainWindow::clearPointCloud()
{
    if (cloudActor && pointCloudVisible) {
        renderer->RemoveActor(cloudActor);
        pointCloudVisible = false;
        vtkWidget->renderWindow()->Render();
    }
}

void MainWindow::loadOBJwithMTL(const QString &objPath, const QString &mtlPath)
{
    QFileInfo objFile(objPath);
    QFileInfo mtlFile(mtlPath);
    qDebug() << "OBJ exists:" << objFile.exists();
    qDebug() << "MTL exists:" << mtlFile.exists();

    if (!objFile.exists()) {
        qWarning() << "OBJ file not found!";
        return;
    }

    vtkNew<vtkOBJImporter> importer;
    importer->SetFileName(objPath.toStdString().c_str());
    if (mtlFile.exists())
        importer->SetFileNameMTL(mtlPath.toStdString().c_str());
    importer->Update();

    vtkRenderer* importerRenderer = importer->GetRenderer();
    if (importerRenderer) {
        vtkActorCollection* actors = importerRenderer->GetActors();
        actors->InitTraversal();
        vtkActor* actor;
        int actorCount = 0;
        while ((actor = actors->GetNextActor())) {
            actor->GetProperty()->SetLighting(true);
            actor->GetProperty()->SetInterpolationToPhong();
            actor->GetProperty()->SetAmbient(0.3);
            actor->GetProperty()->SetDiffuse(0.8);
            renderer->AddActor(actor);
            modelActors.push_back(actor);
            actorCount++;
        }
        qDebug() << "Actors added from OBJ importer:" << actorCount;
    } else {
        qWarning() << "Importer did not create a renderer. Using OBJReader fallback.";
        vtkNew<vtkOBJReader> reader;
        reader->SetFileName(objPath.toStdString().c_str());
        reader->Update();
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());
        vtkSmartPointer<vtkActor> fallbackActor = vtkSmartPointer<vtkActor>::New();
        fallbackActor->SetMapper(mapper);
        fallbackActor->GetProperty()->SetColor(0.7, 0.7, 0.7);
        renderer->AddActor(fallbackActor);
        modelActors.push_back(fallbackActor);
    }
    renderer->ResetCamera();
    vtkWidget->renderWindow()->Render();
}

// ------------------------------------------------------------------
// Constructor & Destructor
// ------------------------------------------------------------------
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , pointCloudVisible(false)
    , texturePlaneActor(nullptr)
{
    ui->setupUi(this);

    vtkWidget = new QVTKOpenGLNativeWidget(this);
    setCentralWidget(vtkWidget);

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

    // Thanh progress bar (thêm vào status bar)
    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    statusBar()->addWidget(progressBar);

    // Toolbar
    QToolBar *toolBar = addToolBar("Reconstruction 3d");
    QAction *load2dAction = toolBar->addAction("1-Load 2D images");
    QAction *load3dAction = toolBar->addAction("1-Load 3D images");
    QAction *loadAction = toolBar->addAction("2-Load multiple 2D images");
    QAction *reconAction = toolBar->addAction("2-Run Reconstruction");
    QAction *showAction = toolBar->addAction("2-Show Point Cloud");
    QAction *clearAction = toolBar->addAction("2-Clear Cloud");

    connect(load2dAction, &QAction::triggered, this, &MainWindow::onLoad2DImages);
    connect(load3dAction, &QAction::triggered, this, &MainWindow::onLoad3DImages);
    connect(loadAction, &QAction::triggered, this, &MainWindow::onLoadMultiple2DImages);
    connect(reconAction, &QAction::triggered, this, &MainWindow::onRunReconstruction);
    connect(showAction, &QAction::triggered, this, &MainWindow::onShowPointCloud);
    connect(clearAction, &QAction::triggered, this, &MainWindow::onClearPointCloud);

    reconstruction = new ReconstructionPipeline();

    // Mặc định load cube (nếu có) - nhưng không bắt buộc, có thể để trống
    QString objPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cube.obj";
    QString mtlPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cube.mtl";
    if (QFileInfo::exists(objPath)) {
        loadOBJwithMTL(objPath, mtlPath);
    } else {
        qDebug() << "Default cube not found, starting with empty scene.";
    }

    // Interactor style
    vtkRenderWindowInteractor* interactor = vtkWidget->renderWindow()->GetInteractor();
    if (!interactor) {
        interactor = vtkRenderWindowInteractor::New();
        vtkWidget->renderWindow()->SetInteractor(interactor);
        interactor->Delete();
    }
    vtkNew<PanStyle> style;
    interactor->SetInteractorStyle(style);
    interactor->Initialize();

    renderer->ResetCamera();
}

MainWindow::~MainWindow()
{
    delete reconstruction;
    delete ui;
}

// ------------------------------------------------------------------
// Slots
// ------------------------------------------------------------------
void MainWindow::onLoad2DImages()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Select 2D Image", "",
                                                    "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileName.isEmpty()) return;

    // Xóa tất cả các đối tượng hiện có: 3D model, point cloud, texture cũ
    clear3DModel();
    clearPointCloud();
    clear2DTexture();

    // Tạo texture plane mới
    vtkSmartPointer<vtkImageReader2> reader;
    if (fileName.endsWith(".png", Qt::CaseInsensitive)) {
        reader = vtkSmartPointer<vtkPNGReader>::New();
    } else {
        reader = vtkSmartPointer<vtkJPEGReader>::New();
    }
    reader->SetFileName(fileName.toStdString().c_str());
    reader->Update();

    vtkImageData* imageData = reader->GetOutput();
    int* dims = imageData->GetDimensions();
    int width = dims[0];
    int height = dims[1];
    double aspect = (double)width / (double)height;

    double planeWidth = aspect;
    double planeHeight = 1.0;
    vtkNew<vtkPlaneSource> plane;
    plane->SetOrigin(-planeWidth/2.0, -planeHeight/2.0, 0.0);
    plane->SetPoint1( planeWidth/2.0, -planeHeight/2.0, 0.0);
    plane->SetPoint2(-planeWidth/2.0,  planeHeight/2.0, 0.0);
    plane->SetResolution(1, 1);
    plane->Update();

    vtkNew<vtkTexture> texture;
    texture->SetInputData(imageData);
    texture->InterpolateOn();

    vtkNew<vtkPolyDataMapper> planeMapper;
    planeMapper->SetInputConnection(plane->GetOutputPort());

    texturePlaneActor = vtkSmartPointer<vtkActor>::New();
    texturePlaneActor->SetMapper(planeMapper);
    texturePlaneActor->SetTexture(texture);
    texturePlaneActor->GetProperty()->SetLighting(false);

    renderer->AddActor(texturePlaneActor);
    renderer->ResetCamera();
    vtkWidget->renderWindow()->Render();

    QMessageBox::information(this, "Info", QString("Đã hiển thị ảnh: %1").arg(fileName));
}

// onLoad3DImages
void MainWindow::onLoad3DImages()
{
    QString objFileName = QFileDialog::getOpenFileName(this, "Select OBJ file", "", "OBJ Files (*.obj)");
    if (objFileName.isEmpty()) return;

    QFileInfo objInfo(objFileName);
    QString mtlFileName = objInfo.path() + "/" + objInfo.completeBaseName() + ".mtl";

    progressBar->setRange(0, 100);
    progressBar->setValue(0);
    progressBar->setVisible(true);
    statusBar()->showMessage("Đang tải mô hình 3D...");
    QApplication::processEvents();

    QTimer *timer = new QTimer(this);
    int current = 0;
    connect(timer, &QTimer::timeout, [&]() {
        if (current < 90) {
            current += 5;
            progressBar->setValue(current);
        }
        QApplication::processEvents();
    });
    timer->start(50);

    clear3DModel();
    clearPointCloud();
    clear2DTexture();
    loadOBJwithMTL(objFileName, mtlFileName);

    timer->stop();
    delete timer;
    progressBar->setValue(100);
    QApplication::processEvents();
    QThread::msleep(200);
    progressBar->setVisible(false);
    statusBar()->clearMessage();

    QMessageBox::information(this, "Info", QString("Đã tải mô hình: %1").arg(objFileName));
}

// ─────────────────────────────────────────────────────────────────────────────
// Thay thế hàm onLoadMultiple2DImages() trong mainwindow.cpp
// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::onLoadMultiple2DImages()
{
    QStringList files = QFileDialog::getOpenFileNames(
        this, "Select Images", "", "Images (*.png *.jpg *.bmp)");
    if (files.isEmpty()) return;

    // ✅ Sắp xếp đúng thứ tự tên file (templeR0001 → templeR0047)
    files.sort();

    std::vector<QString> paths;
    for (const auto &f : files) paths.push_back(f);
    reconstruction->setImages(paths);

    // ✅ Tự động tìm file *_par.txt trong cùng thư mục ảnh
    QFileInfo firstFile(files.first());
    QString folder = firstFile.absolutePath();

    // Các tên file params phổ biến của Middlebury dataset
    QStringList paramsNames = {
        "temple_par.txt",
        "templeR_par.txt",
        "templeRing_par.txt",
        "dinoR_par.txt",
        "dinoRing_par.txt",
        "par.txt",
        "camera_params.txt"
    };

    bool paramsLoaded = false;
    QString loadedParamsPath;
    for (const auto &name : paramsNames) {
        QString paramsPath = folder + "/" + name;
        if (QFileInfo::exists(paramsPath)) {
            paramsLoaded = reconstruction->loadCameraParams(paramsPath);
            if (paramsLoaded) {
                loadedParamsPath = paramsPath;
                break;
            }
        }
    }

    // Thông báo kết quả
    QString msg = QString("Đã tải %1 ảnh.").arg(paths.size());
    if (paramsLoaded) {
        msg += QString("\n\n✅ Camera params: %1").arg(
            QFileInfo(loadedParamsPath).fileName());
        msg += "\n→ Sẽ dùng ground-truth projection matrices.";
    } else {
        msg += "\n\n⚠️ Không tìm thấy file *_par.txt trong:\n" + folder;
        msg += "\n→ Đặt file templeR_par.txt cùng thư mục ảnh để có kết quả tốt nhất.";
        msg += "\n→ Sẽ dùng estimated pose (kém chính xác hơn).";
    }

    QMessageBox::information(this, "Load Images", msg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Thay thế hàm onRunReconstruction() trong mainwindow.cpp
// ─────────────────────────────────────────────────────────────────────────────
void MainWindow::onRunReconstruction()
{
    progressBar->setRange(0, 0);   // chế độ "đang chạy" (animated)
    progressBar->setVisible(true);
    statusBar()->showMessage("Đang tái tạo 3D...");
    QApplication::processEvents();

    bool success = reconstruction->reconstruct();

    progressBar->setRange(0, 100);
    progressBar->setValue(100);
    QApplication::processEvents();
    QThread::msleep(200);
    progressBar->setVisible(false);
    statusBar()->clearMessage();

    if (!success) {
        QMessageBox::warning(this, "Lỗi",
                             "Reconstruction thất bại!\n"
                             "Kiểm tra:\n"
                             "  • Đã load ít nhất 2 ảnh chưa?\n"
                             "  • File templeR_par.txt có cùng thư mục ảnh không?");
        return;
    }

    // ✅ Tự động hiển thị point cloud ngay sau khi reconstruct
    onShowPointCloud();

    QMessageBox::information(this, "Thành công",
                             QString("Reconstruction hoàn tất!\nTổng số điểm 3D: %1")
                                 .arg(reconstruction->getPointCloud().size()));
}

void MainWindow::onShowPointCloud()
{
    auto pts = reconstruction->getPointCloud();
    auto colors = reconstruction->getPointColors();
    if (pts.empty()) {
        QMessageBox::warning(this, "Warning", "Chưa có point cloud nào. Hãy chạy Reconstruction trước.");
        return;
    }

    qDebug() << "Showing point cloud: points=" << pts.size() << ", colors=" << colors.size();

    // Xóa point cloud cũ nếu có
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
}

void MainWindow::onClearPointCloud()
{
    clearPointCloud();
}
