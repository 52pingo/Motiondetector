cmake_minimum_required(VERSION 3.16)
project(MotionDetectorQt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

add_executable(MotionDetectorQt
    main.cpp
    mainwindow.cpp
    mainwindow.h
    motion_detector.cpp
    motion_detector.h
    app.rc
)

# Windows 应用图标
set_target_properties(MotionDetectorQt PROPERTIES
    WIN32_EXECUTABLE TRUE
)

target_include_directories(MotionDetectorQt PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(MotionDetectorQt PRIVATE
    Qt6::Widgets
    ${OpenCV_LIBS}
)
