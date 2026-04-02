cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES CXX CUDA)

# FORZAR ARQUITECTURA
set(CMAKE_CUDA_ARCHITECTURES 89)
set(TCNN_CUDA_ARCHITECTURES 89)

# ----------------------------
# C++ / CUDA standard
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ----------------------------
# CUDA settings
# ----------------------------
find_program(GXX11_COMPILER g++-11)
if(GXX11_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${GXX11_COMPILER})
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_AMXTILEINTRIN_H_INCLUDED")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_AMXINT8INTRIN_H_INCLUDED")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_AMXBF16INTRIN_H_INCLUDED")

string(REPLACE "-Wpedantic" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wno-pedantic")

# ----------------------------
# tiny-cuda-nn
# ----------------------------
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Añadimos el directorio de TCNN
set(TCNN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tiny-cuda-nn")
add_subdirectory("${TCNN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/tiny-cuda-nn-build")

# Suprimir warnings de terceros en tiny-cuda-nn
set_target_properties(tiny-cuda-nn PROPERTIES
    COMPILE_FLAGS "-Wno-stringop-overread"
)

# ----------------------------
# Librerías externas
# ----------------------------
find_package(PNG REQUIRED)
find_package(assimp REQUIRED)
find_package(SDL2 REQUIRED)

# ----------------------------
# Ejecutable
# ----------------------------
add_executable(raytracer
    main.cu
    bvh.cu
    mlp.cu
    tinybvh_wrapper.cpp
)

# ----------------------------
# Enlace y Propiedades
# ----------------------------
target_link_libraries(raytracer
    PRIVATE
    tiny-cuda-nn
    PNG::PNG
    assimp::assimp
    ${SDL2_LIBRARIES}
)

target_include_directories(raytracer PRIVATE
    ${PNG_INCLUDE_DIRS}
    ${ASSIMP_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/../tinybvh-main"
    "${TCNN_SOURCE_DIR}/include"
)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)