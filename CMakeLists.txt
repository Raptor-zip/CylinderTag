cmake_minimum_required(VERSION 3.16)
project(CylinderTag VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Add headers directory
include_directories(${PROJECT_SOURCE_DIR}/header)

# Add source files
add_library(cylindertag_core
    CylinderTag.cpp
    corner_detector.cpp
    pose_estimation.cpp
)

# Link libraries
target_link_libraries(cylindertag_core PUBLIC ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_include_directories(cylindertag_core PUBLIC ${PROJECT_SOURCE_DIR})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} cylindertag_core)

# Headless detector for recorded videos (no camera calibration / pose required).
add_executable(cylindertag_video
    video_detect.cpp
)
target_link_libraries(cylindertag_video cylindertag_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(cylindertag_decoder_test tests/decoder_test.cpp)
    target_link_libraries(cylindertag_decoder_test cylindertag_core)
    add_test(NAME dictionary_decoder COMMAND cylindertag_decoder_test ${PROJECT_SOURCE_DIR}/CTag_2f12c.marker)
endif()
