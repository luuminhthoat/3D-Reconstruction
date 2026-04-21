#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Tạo QVTKOpenGLNativeWidget làm central widget
    QVTKOpenGLNativeWidget* vtkWidget = new QVTKOpenGLNativeWidget(this);
    setCentralWidget(vtkWidget);

    // Tạo nguồn hình học: hình trụ
    vtkNew<vtkCylinderSource> cylinder;
    cylinder->SetHeight(3.0);
    cylinder->SetRadius(1.0);
    cylinder->SetResolution(32);

    // Mapper
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(cylinder->GetOutputPort());

    // Actor
    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);

    // Renderer
    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.2, 0.4);

    // Gán renderer vào VTK widget
    vtkWidget->renderWindow()->AddRenderer(renderer);
}

MainWindow::~MainWindow()
{
    delete ui;
}