#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "reconstructionpipeline.h"
#include <QVBoxLayout>
#include <QToolBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QDebug>
#include <vtkOBJReader.h>
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

    // Load OBJ mặc định
    vtkNew<vtkOBJReader> reader;
    reader->SetFileName("C:/Users/ADMIN/Documents/3D-Reconstruction/3DModels/Audi_R8_2017.obj");
    reader->Update();

    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(reader->GetOutputPort());

    objActor = vtkSmartPointer<vtkActor>::New();
    objActor->SetMapper(mapper);
    renderer->AddActor(objActor);
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

    // Xóa cloud cũ nếu có
    onClearPointCloud();

    // Tạo vtkPoints và vtkPolyData
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
            // colors[i] là cv::Vec3b (B,G,R). VTK cần (R,G,B)
            vtkColors->InsertNextTuple3(colors[i][2], colors[i][1], colors[i][0]);
        } else {
            // Mặc định màu trắng
            vtkColors->InsertNextTuple3(255, 255, 255);
        }
    }

    vtkNew<vtkPolyData> polyData;
    polyData->SetPoints(vtkPoints);
    polyData->SetVerts(vertices);
    polyData->GetPointData()->SetScalars(vtkColors);

    // Sử dụng vtkVertexGlyphFilter để tạo các điểm
    vtkNew<vtkVertexGlyphFilter> glyphFilter;
    glyphFilter->SetInputData(polyData);
    glyphFilter->Update();

    vtkNew<vtkPolyDataMapper> cloudMapper;
    cloudMapper->SetInputConnection(glyphFilter->GetOutputPort());

    cloudActor = vtkSmartPointer<vtkActor>::New();
    cloudActor->SetMapper(cloudMapper);
    // Tăng kích thước điểm để nhìn rõ màu
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
