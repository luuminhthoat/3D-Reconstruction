#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QVBoxLayout>
#include "PanStyle.h"  // Thay vì include vtkInteractorStyleTrackballCamera
#include <vtkOBJReader.h> // Thêm thư viện để đọc file OBJ
#include <vtkOBJImporter.h> // Thêm thư viện để import model OBJ có texture

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QVTKOpenGLNativeWidget* vtkWidget = new QVTKOpenGLNativeWidget(this);
    vtkWidget->setContextMenuPolicy(Qt::NoContextMenu);
    setCentralWidget(vtkWidget);

    //vtkNew<vtkCylinderSource> cylinder;
    //cylinder->SetHeight(3.0);
    //cylinder->SetRadius(1.0);
    //cylinder->SetResolution(32);

    //vtkNew<vtkPolyDataMapper> mapper;
    //mapper->SetInputConnection(cylinder->GetOutputPort());

    // Load file OBJ
    vtkNew<vtkOBJReader> reader;
    reader->SetFileName("C:/Users/ADMIN/Documents/3D-Reconstruction/FinalBaseMesh.obj");

    reader->Update();

    // Mapper
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkNew<vtkActor> actor;
    actor->SetMapper(mapper);

    vtkNew<vtkRenderer> renderer;
    renderer->AddActor(actor);
    renderer->SetBackground(0.1, 0.2, 0.4);

    // Fit camera
    renderer->ResetCamera();

    vtkWidget->renderWindow()->AddRenderer(renderer);

    // Lấy interactor
    vtkRenderWindowInteractor* interactor = vtkWidget->renderWindow()->GetInteractor();
    if (!interactor)
    {
        interactor = vtkRenderWindowInteractor::New();
        vtkWidget->renderWindow()->SetInteractor(interactor);
        interactor->Delete();
    }

    // Gán custom style
    vtkNew<PanStyle> style;
    interactor->SetInteractorStyle(style);

    interactor->Initialize();
}

MainWindow::~MainWindow()
{
    delete ui;
}