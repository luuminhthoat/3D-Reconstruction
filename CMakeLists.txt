cmake_minimum_required(VERSION 3.19)
project(3D-Reconstruction LANGUAGES CXX)

# 1. Tìm Qt6
find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets OpenGL)

qt_standard_project_setup()

# 2. Tìm VTK (tắt Vulkan)
set(VTK_MODULE_ENABLE_VTK_RenderingVulkan NO)
set(VTK_DIR "C:/Program Files/VTK/lib/cmake/vtk-9.6" CACHE PATH "Path to VTK")

find_package(VTK REQUIRED COMPONENTS
    CommonCore
    RenderingOpenGL2
    GUISupportQt
    ViewsQt
)

# 3. Khai báo executable (PHẢI ĐẶT TRƯỚC KHI GỌI target_...)
qt_add_executable(3D-Reconstruction
    WIN32 MACOSX_BUNDLE
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    PanStyle.h PanStyle.cpp
)

# 4. Thêm include và định nghĩa cho target (SAU KHI target đã tồn tại)
target_include_directories(3D-Reconstruction PRIVATE ${VTK_INCLUDE_DIRS})
target_compile_definitions(3D-Reconstruction PRIVATE ${VTK_DEFINITIONS})

# 5. Liên kết thư viện
target_link_libraries(3D-Reconstruction
    PRIVATE
        Qt6::Core      # chú ý: Qt6::Core, không phải QT::Core
        Qt6::Widgets
        Qt6::OpenGL
        ${VTK_LIBRARIES}
)

# 6. Cài đặt
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
