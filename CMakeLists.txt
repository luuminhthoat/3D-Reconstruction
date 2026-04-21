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

# 4. Eigen3 (header-only, không cần find_package)
set(EIGEN3_INCLUDE_DIR "C:/eigen-5.0.0")
if(NOT EXISTS ${EIGEN3_INCLUDE_DIR}/Eigen/Core)
    message(FATAL_ERROR "Eigen3 not found at ${EIGEN3_INCLUDE_DIR}")
endif()

# 5. Khai báo executable (thêm các file mới)
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
)

# 6. Thêm include directories cho target
target_include_directories(3D-Reconstruction PRIVATE
    ${VTK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}      # Thay vì ${EIGEN3_INCLUDE_DIRS}
)

# 7. Định nghĩa compile flags từ VTK
target_compile_definitions(3D-Reconstruction PRIVATE ${VTK_DEFINITIONS})

# 8. Liên kết thư viện (bỏ Eigen3::Eigen)
target_link_libraries(3D-Reconstruction
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::OpenGL
        ${VTK_LIBRARIES}
        ${OpenCV_LIBS}
        # Eigen3::Eigen không cần vì chỉ là header
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
