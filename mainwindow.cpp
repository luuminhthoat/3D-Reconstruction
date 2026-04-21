#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "reconstructionpipeline.h"
#include <QVBoxLayout>
#include <QToolBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <QFileInfo>
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
#include "PanStyle.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , pointCloudVisible(false)
{
    ui->setupUi(this);

    vtkWidget = new QVTKOpenGLNativeWidget(this);
    setCentralWidget(vtkWidget);

    renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.1, 0.2, 0.4);
    vtkWidget->renderWindow()->AddRenderer(renderer);

    // Thêm đèn headlight để chiếu sáng từ camera
    vtkSmartPointer<vtkLight> headlight = vtkSmartPointer<vtkLight>::New();
    headlight->SetLightTypeToHeadlight();
    headlight->SetIntensity(1.5);
    renderer->AddLight(headlight);

    // Thêm các đèn phụ
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

    // Load OBJ + MTL
    QString objPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cottage_obj.obj";
    QString mtlPath = "C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/85-cottage_obj/cottage_obj.mtl";

    QFileInfo objFile(objPath);
    QFileInfo mtlFile(mtlPath);
    qDebug() << "OBJ exists:" << objFile.exists();
    qDebug() << "MTL exists:" << mtlFile.exists();

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
            // In số polygons và màu gốc từ material
            vtkPolyData* polyData = vtkPolyData::SafeDownCast(actor->GetMapper()->GetInput());
            if (polyData) {
                vtkIdType numPolys = polyData->GetNumberOfPolys();
                qDebug() << "Actor" << actorCount << "has" << numPolys << "polygons";
            } else {
                qDebug() << "Actor" << actorCount << "has no poly data";
            }

            // Lấy màu đã được importer đọc từ MTL
            double* color = actor->GetProperty()->GetColor();
            qDebug() << "Material color from MTL (R,G,B):" << color[0] << color[1] << color[2];

            // Bật lighting để màu hiển thị đúng (không ghi đè)
            actor->GetProperty()->SetLighting(true);
            actor->GetProperty()->SetInterpolationToPhong();
            actor->GetProperty()->SetAmbient(0.3);
            actor->GetProperty()->SetDiffuse(0.8);

            renderer->AddActor(actor);
            actorCount++;
        }
        qDebug() << "Actors added from OBJ importer:" << actorCount;
        if (actorCount == 0) {
            // fallback...
            vtkNew<vtkOBJReader> reader;
            reader->SetFileName(objPath.toStdString().c_str());
            reader->Update();
            vtkNew<vtkPolyDataMapper> mapper;
            mapper->SetInputConnection(reader->GetOutputPort());
            objActor = vtkSmartPointer<vtkActor>::New();
            objActor->SetMapper(mapper);
            objActor->GetProperty()->SetColor(0.7, 0.7, 0.7);
            renderer->AddActor(objActor);
        }
    } else {
        // fallback...
        vtkNew<vtkOBJReader> reader;
        reader->SetFileName(objPath.toStdString().c_str());
        reader->Update();
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputConnection(reader->GetOutputPort());
        objActor = vtkSmartPointer<vtkActor>::New();
        objActor->SetMapper(mapper);
        objActor->GetProperty()->SetColor(0.7, 0.7, 0.7);
        renderer->AddActor(objActor);
    }

    renderer->ResetCamera();

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

    // Toolbar
    QToolBar *toolBar = addToolBar("Reconstruction");
    QAction *loadAction = toolBar->addAction("Load Images");
    QAction *reconAction = toolBar->addAction("Run Reconstruction");
    QAction *showAction = toolBar->addAction("Show Point Cloud");
    QAction *clearAction = toolBar->addAction("Clear Cloud");

    connect(loadAction, &QAction::triggered, this, &MainWindow::onLoadImages);
    connect(reconAction, &QAction::triggered, this, &MainWindow::onRunReconstruction);
    connect(showAction, &QAction::triggered, this, &MainWindow::onShowPointCloud);
    connect(clearAction, &QAction::triggered, this, &MainWindow::onClearPointCloud);

    reconstruction = new ReconstructionPipeline();
}

MainWindow::~MainWindow()
{
    delete reconstruction;
    delete ui;
}

void MainWindow::onLoadImages()
{
    QStringList files = QFileDialog::getOpenFileNames(this, "Select Images", "",
                                                      "Images (*.png *.jpg *.bmp)");
    if (files.isEmpty()) return;

    std::vector<QString> paths;
    for (const auto &f : files) paths.push_back(f);
    reconstruction->setImages(paths);
    QMessageBox::information(this, "Info", QString("Đã tải %1 ảnh").arg(paths.size()));
}

void MainWindow::onRunReconstruction()
{
    if (!reconstruction->reconstruct()) {
        QMessageBox::warning(this, "Error", "Reconstruction thất bại! Kiểm tra số lượng ảnh hoặc chất lượng features.");
        return;
    }
    QMessageBox::information(this, "Success", "Point cloud đã được tạo.");
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

    onClearPointCloud();

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
    if (cloudActor && pointCloudVisible) {
        renderer->RemoveActor(cloudActor);
        vtkWidget->renderWindow()->Render();
        pointCloudVisible = false;
    }
}
