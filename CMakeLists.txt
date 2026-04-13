cmake_minimum_required(VERSION 3.18)

project(raytracer LANGUAGES C CXX CUDA)

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
find_package(OpenGL REQUIRED)

# GLFW instalado en el sistema
set(GLFW_TARGET "")
find_package(glfw3 CONFIG QUIET)

if(TARGET glfw)
    set(GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(GLFW_TARGET glfw3::glfw)
else()
    set(GLFW_LOCAL_SRC "${CMAKE_BINARY_DIR}/_deps/glfw-src")
    set(GLFW_LOCAL_BUILD "${CMAKE_BINARY_DIR}/_deps/glfw-build")

    if(EXISTS "${GLFW_LOCAL_SRC}/CMakeLists.txt")
        message(STATUS "Usando GLFW local en ${GLFW_LOCAL_SRC}")
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        add_subdirectory("${GLFW_LOCAL_SRC}" "${GLFW_LOCAL_BUILD}")
        set(GLFW_TARGET glfw)
    else()
        message(FATAL_ERROR
            "No se encontró GLFW3 ni por CMake ni en build/_deps. "
            "Instala libglfw3-dev o restaura la carpeta build/_deps/glfw-src.")
    endif()
endif()

# ----------------------------
# Ejecutable
# ----------------------------
add_executable(raytracer
    main.cpp
    bvh.cpp
    mlp.cpp
    RenderKernel.cu
    NrcKernel.cu
    tinybvh_wrapper.cpp
)

add_library(raytracer_ui STATIC
    GuiLayer.cpp
    glad.c
    imgui_src/imgui.cpp
    imgui_src/imgui_draw.cpp
    imgui_src/imgui_tables.cpp
    imgui_src/imgui_widgets.cpp
    imgui_src/imgui_impl_glfw.cpp
    imgui_src/imgui_impl_opengl3.cpp
)

set_source_files_properties(main.cpp mlp.cpp bvh.cpp PROPERTIES LANGUAGE CUDA)

# ----------------------------
# Enlace y Propiedades
# ----------------------------
target_link_libraries(raytracer
    PRIVATE
    tiny-cuda-nn
    PNG::PNG
    assimp::assimp
    raytracer_ui
)

target_compile_definitions(raytracer_ui PRIVATE IMGUI_IMPL_OPENGL_LOADER_CUSTOM)

target_link_libraries(raytracer_ui
    PUBLIC
    OpenGL::GL
    ${GLFW_TARGET}
)

target_include_directories(raytracer PRIVATE
    ${PNG_INCLUDE_DIRS}
    ${ASSIMP_INCLUDE_DIRS}
    "${TCNN_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui_src"
)

target_include_directories(raytracer_ui PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/imgui"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui_src"
)

set_target_properties(raytracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)