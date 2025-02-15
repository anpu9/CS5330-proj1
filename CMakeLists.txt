cmake_minimum_required(VERSION 3.26)
project(Project1)

set(CMAKE_CXX_STANDARD 17)

# Base packages
find_package(OpenCV REQUIRED)
find_package(JsonCpp REQUIRED)
find_package(PCL 1.3 REQUIRED COMPONENTS common io visualization)

# ONNX Runtime setup
set(ONNX_RUNTIME_DIR "/opt/onnxruntime/onnxruntime-osx-arm64-1.20.1")
find_path(ONNX_RUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ONNX_RUNTIME_DIR}/include
        REQUIRED
)
find_library(ONNX_RUNTIME_LIB
        NAMES onnxruntime
        HINTS ${ONNX_RUNTIME_DIR}/lib
        REQUIRED
)

# Function for OpenCV-only executables
function(add_opencv_executable exe_name)
    add_executable(${exe_name} ${ARGN})
    target_link_libraries(${exe_name} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${exe_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
endfunction()

# Function for OpenCV + PCL executables
function(add_opencv_pcl_executable exe_name)
    add_executable(${exe_name} ${ARGN})
    target_link_libraries(${exe_name} PRIVATE
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES}
    )
    target_include_directories(${exe_name} PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS}
    )
    target_compile_definitions(${exe_name} PRIVATE ${PCL_DEFINITIONS})
endfunction()

# Function for executables with all dependencies (OpenCV + PCL + ONNX + JsonCpp)
function(add_full_executable exe_name)
    add_executable(${exe_name} ${ARGN})
    target_link_libraries(${exe_name} PRIVATE
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES}
            ${ONNX_RUNTIME_LIB}
            JsonCpp::JsonCpp
    )
    target_include_directories(${exe_name} PRIVATE
            ${OpenCV_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS}
            ${ONNX_RUNTIME_INCLUDE_DIR}
            JsonCpp::JsonCpp
    )
    target_compile_definitions(${exe_name} PRIVATE ${PCL_DEFINITIONS})
endfunction()

# Add OpenCV-only executables
add_opencv_executable(Filter src/filterDisplay.cpp)
#add_opencv_executable(ImageDisplay src/imgDisplay.cpp)
# Add OpenCV-only executables
add_full_executable(VidDisplay
        src/vidDisplay.cpp
        src/filter.cpp
        src/pcl3d.cpp
)

# Add executables with all dependencies
add_full_executable(da2-video
        src/da2-video.cpp
        src/pcl3d.cpp
)
add_full_executable(da2-example
        src/da3d.cpp
        src/pcl3d.cpp
)

# PCL and JsonCpp link directories
link_directories(${PCL_LIBRARY_DIRS})
if(JSONCPP_LIBRARY_DIRS)
    link_directories(${JSONCPP_LIBRARY_DIRS})
endif()