#pragma once

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkImageActor.h>
#include <vtkVolume.h>
#include <vtkAlgorithmOutput.h>
#include <QString>

class DicomLoader {
public:
    // Nạp toàn bộ series DICOM từ thư mục
    static vtkSmartPointer<vtkImageData> loadSeries(const QString &dirPath);

    // Tạo Actor hiển thị lát cắt (MPR)
    static vtkSmartPointer<vtkImageActor> createSlice(vtkImageData* data, int orientation, double window, double level);

    // Tạo Volume hiển thị khối 3D
    static vtkSmartPointer<vtkVolume> createVolume(vtkImageData* data, double range[2]);
};
