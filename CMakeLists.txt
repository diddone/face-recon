cmake_minimum_required(VERSION 3.10)
project(bfm_manager)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-lboost_program_options")
if(UNIX)
   add_compile_options(-fPIC)
endif()

# Debug mode specific settings
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fsanitize=address -fsanitize=undefined")
endif()

# if(APPLE)
# if(CMAKE_C_COMPILER_ID MATCHES "Clang")
#     set(OpenMP_C "${CMAKE_C_COMPILER}" CACHE STRING "" FORCE)
#     set(OpenMP_C_FLAGS "-Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#     set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
#     set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
#     set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
#     set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES} CACHE STRING "" FORCE)
# endif()
# if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   set(OpenMP_CXX "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
#   set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument" CACHE STRING "" FORCE)
#   set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5" CACHE STRING "" FORCE)
#   set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
#   set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
#   set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES} CACHE STRING "" FORCE)
# endif()
# endif()

add_library(bfm_manager SHARED src/bfm_manager.cpp
        src/visualizer.h)
target_compile_features(bfm_manager PRIVATE cxx_std_17)

###############################################################################################
######################################## Dedendency ###########################################
###############################################################################################

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra -Wundef -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse -funroll-loops -Wno-reorder")
endif()

#VTK
find_package(VTK)

# Eigen3
MESSAGE(STATUS "FINDING EIGEN")
find_package(Eigen3 REQUIRED NO_MODULE)
target_include_directories(bfm_manager PUBLIC ${EIGEN3_INCLUDE_DIR})

# Hdf5
find_package(HDF5 REQUIRED C CXX)
target_include_directories(bfm_manager PUBLIC ${HDF5_INCLUDE_DIRS})

# OpenCV
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")

find_package(OpenCV REQUIRED)
target_include_directories(bfm_manager PUBLIC ${OpenCV_INCLUDE_DIRS})

# Glog
find_package(glog REQUIRED)
find_path(GLOG_INCLUDE_DIR NAMES raw_logging.h
   PATHS
         /usr/include/glog
         /usr/local/include/glog
)
set(GLOG_INCLUDE_DIRS ${GLOG_INCLUDE_DIR})
target_include_directories(bfm_manager PUBLIC ${GLOG_INCLUDE_DIR})

# Set the OpenMP paths manually
if (APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    set(OpenMP_CXX_FLAGS "${OpenMP_C_FLAGS}")
    set(OpenMP_CXX_LIB_NAMES "${OpenMP_C_LIB_NAMES}") # Manually specify the C++ library names
    set(OpenMP_CXX_LIBRARIES "${OpenMP_omp_LIBRARY}")

    # Since Homebrew uses symlinks in /usr/local/opt, use these paths:
    include_directories("/usr/local/opt/libomp/include")
    link_directories("/usr/local/opt/libomp/lib")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
   target_link_libraries(bfm_manager PUBLIC OpenMP::OpenMP_CXX)
else()
   message(FATAL_ERROR "OpenMP not found. Please make sure OpenMP is installed on your system.")
endif()



find_package(yaml-cpp REQUIRED)
#Yaml
# set(YAML_BUILD_SHARED_LIBS ON)
# MESSAGE(STATUS "FINDING YAML")
# include(FetchContent)
# FetchContent_Declare(
#     yaml-cpp
#     GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
#     # GIT_TAG <tag_name> # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
# )
# FetchContent_GetProperties(yaml-cpp)
# if(NOT yaml-cpp_POPULATED)
#     message(STATUS "Fetching yaml-cpp...")
#     FetchContent_Populate(yaml-cpp)
#     add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
# endif()

#GLFW
find_package(glfw3 REQUIRED)
add_subdirectory("./include/glad")


# #PCL
MESSAGE(STATUS "FOUNDING PCL")
find_package(PCL REQUIRED)
target_include_directories(bfm_manager PUBLIC ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_directories(bfm_manager PUBLIC ${PCL_LIBRARY_DIRS})

if(UNIX)
    set(LIBS
        /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.65.1
        /usr/lib/x86_64-linux-gnu/libboost_system.so.1.65.1
    )
endif()

#Boost
find_package(Boost REQUIRED system filesystem program_options)
target_include_directories(bfm_manager PUBLIC ${Boost_INCLUDE_DIRS})

#Ceres
find_package(Ceres REQUIRED)

# Find Dlib package
find_package(dlib REQUIRED)

# Bfm Manager
target_include_directories(bfm_manager PUBLIC ./include)
target_link_libraries(bfm_manager PUBLIC
                    m
                    Eigen3::Eigen
                    ${OpenCV_LIBS}
                    ${HDF5_LIBRARIES}
                    ${HDF5_CXX_LIBRARIES}
                    ${Boost_LIBRARIES}
                    glog::glog
                    Boost::filesystem
                    ${OpenMP_CXX_LIBRARIES})

add_executable(main src/main.cpp src/utils.h include/glad/src/glad.c
src/shader_class.h include/stb_image.h src/utils.h
src/face_mesh.h
src/visualizer.h
src/optimizer_class.h)
target_include_directories(main PUBLIC ./include ${DLIB_INCLUDE_DIRECTORY})
target_compile_features(main PRIVATE cxx_std_17)
target_link_libraries(main glfw glad bfm_manager yaml-cpp dlib ${PCL_LIBRARIES} Ceres::ceres)
target_compile_features(main PRIVATE cxx_std_17)

# Visualizer
add_executable(visualizer src/visualizer.cpp include/glad/src/glad.c
        src/shader_class.h include/stb_image.h src/utils.h
        src/face_mesh.h
        src/visualizer.h
        src/optimizer_class.h)
target_include_directories(visualizer PUBLIC ./include)
target_link_libraries(visualizer glfw glad bfm_manager yaml-cpp Boost::filesystem Boost::system)
target_include_directories(visualizer PUBLIC ./include)
target_compile_features(visualizer PRIVATE cxx_std_17)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# del
add_executable(del src/del.cpp src/utils.h)
target_link_libraries(del bfm_manager
                      Boost::program_options)
###############################################################################################
######################################## Options ##############################################
###############################################################################################


# OPTION(GEN_BFM_EXAMPLES "Generate usage examples of Basel Face Moldel manager." OFF)
# if(GEN_BFM_EXAMPLES)
#    add_executable(gen_faces example/ex_gen_faces.cpp)
#    target_link_libraries(gen_faces bfm_manager)
# endif()