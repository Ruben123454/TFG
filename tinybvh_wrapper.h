// bvh_wrapper.h
#ifndef BVH_WRAPPER_H
#define BVH_WRAPPER_H


#include "primitiva.h"
#include <vector>
#ifndef __CUDACC__
#define CWBVH_COMPRESSED_TRIS
#include "../tinybvh-main/tiny_bvh.h"
#endif


// Estructura que contiene los punteros que enviaremos a la GPU
struct TinyBVHD_GPU {
    float4* d_nodos = nullptr; // Nodos CWBVH
    float4* d_tris = nullptr; // Triangulos procesados

    unsigned int num_nodos;
    unsigned int num_tris;
};

class TinyBVH {
public:
    TinyBVH();
    ~TinyBVH();

    void construirBVH(const std::vector<Primitiva>& primitivas_host);
    void obtenerInfo();
    TinyBVHD_GPU obtenerDatosGPU();
    const TinyBVHD_GPU* getNodosGPU();
    const Primitiva* getPrimitivasGPU();
    unsigned int getNumNodos();
    unsigned int getNumPrimitivas();

private:
    TinyBVHD_GPU datos_gpu;
    void* cwbvh_ptr;

    float4* d_nodos = nullptr;
    float4* d_tris = nullptr;

    Primitiva* primitivas_gpu;
    unsigned int num_primitivas_gpu = 0;
};

#endif