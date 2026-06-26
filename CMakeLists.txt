cmake_minimum_required(VERSION 3.10)
project(MotionDetection VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 可执行文件
add_executable(motion_detect
    src/main.cpp
    src/motion_detector.cpp
)

# 包含头文件目录
target_include_directories(motion_detect PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# 链接 OpenCV
target_link_libraries(motion_detect ${OpenCV_LIBS})

# 编译选项
if(MSVC)
    target_compile_options(motion_detect PRIVATE /W4 /O2)
else()
    target_compile_options(motion_detect PRIVATE -Wall -Wextra -O2)
endif()
