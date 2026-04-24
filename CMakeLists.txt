cmake_minimum_required(VERSION 3.19)
project(3D-Reconstruction LANGUAGES CXX)

# 1. Tìm Qt6
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets OpenGL)
qt_standard_project_setup()

# 2. Tìm VTK (tắt Vulkan)
set(VTK_MODULE_ENABLE_VTK_RenderingVulkan NO)
set(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6" CACHE PATH "Path to VTK")

# Bổ sung thêm module vtkFiltersCore để dùng vtkVertexGlyphFilter cho point cloud
find_package(VTK REQUIRED COMPONENTS
    CommonCore
    RenderingOpenGL2
    GUISupportQt
    ViewsQt
    IOGeometry
    IOImport
    FiltersCore
)

# 3. Tìm OpenCV (cần cho phase 2)
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)

# 3.5. Tìm PCL (Point Cloud Library)
set(PCL_DIR "C:/Program Files/PCL 1.15.1/cmake" CACHE PATH "Path to PCL CMake files")
find_package(PCL REQUIRED COMPONENTS common filters io)

# 4. Eigen3 sẽ tự động được include từ PCL_INCLUDE_DIRS (PCL thường đi kèm Eigen 3.x)

# 5. Khai báo executable
qt_add_executable(3D-Reconstruction
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    PanStyle.h
    PanStyle.cpp
    reconstructionpipeline.h
    reconstructionpipeline.cpp
    reconstructthread.h
    reconstructthread.cpp
)

# 6. Thêm include directories cho target
target_include_directories(3D-Reconstruction PRIVATE
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}          # <--- Thêm PCL (đã bao gồm Eigen)
)

# 7. Định nghĩa compile flags từ VTK và PCL
target_compile_definitions(3D-Reconstruction PRIVATE
    ${VTK_DEFINITIONS}
    ${PCL_DEFINITIONS}           # <--- Thêm PCL
    PCL_SILENCE_MALLOC_WARNING=1 # <--- Tắt cảnh báo AVX/malloc mismatch
)

# 8. Liên kết thư viện
target_link_libraries(3D-Reconstruction
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}         # <--- Thêm PCL
)

# 9. Cài đặt
include(GNUInstallDirs)
install(TARGETS 3D-Reconstruction
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET 3D-Reconstruction
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
