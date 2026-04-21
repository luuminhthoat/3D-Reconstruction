#include <QApplication>
#include <QSurfaceFormat>
#include <vtkOpenGLRenderWindow.h>
#include "mainwindow.h"

int main(int argc, char* argv[])
{
    // Tắt multisampling để tránh xung đột với Qt
    vtkOpenGLRenderWindow::SetGlobalMaximumNumberOfMultiSamples(0);

    QSurfaceFormat::setDefaultFormat(QSurfaceFormat::defaultFormat());

    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return app.exec();
}