// bvh_wrapper.cpp
#define TINYBVH_IMPLEMENTATION
#define CWBVH_COMPRESSED_TRIS
#include "tinybvh_wrapper.h"

using namespace tinybvh;
using namespace std;

TinyBVH::TinyBVH() {
    cwbvh_ptr = new BVH8_CWBVH();
    d_nodos = nullptr;
    d_tris = nullptr;
    primitivas_gpu = nullptr;

    datos_gpu.d_nodos = nullptr;
    datos_gpu.d_tris = nullptr;
    datos_gpu.num_nodos = 0;
    datos_gpu.num_tris = 0;
    num_primitivas_gpu = 0;
}

TinyBVH::~TinyBVH() {
    delete static_cast<BVH8_CWBVH*>(cwbvh_ptr);

    if (d_nodos) cudaFree(d_nodos);
    if (d_tris) cudaFree(d_tris);
    if (primitivas_gpu) cudaFree(primitivas_gpu);
}

void TinyBVH::construirBVH(const std::vector<Primitiva>& primitivas_host) {
    vector<bvhvec4> vertices_out;
    vertices_out.reserve(primitivas_host.size() * 3);
    vector<Primitiva> triangulos_host;
    triangulos_host.reserve(primitivas_host.size());

    for (size_t i = 0; i < primitivas_host.size(); i++) {
        if (primitivas_host[i].tipo == TRIANGULO) {
            vertices_out.push_back(bvhvec4(primitivas_host[i].triangulo.v0.x, primitivas_host[i].triangulo.v0.y, primitivas_host[i].triangulo.v0.z, 0.0f));
            vertices_out.push_back(bvhvec4(primitivas_host[i].triangulo.v1.x, primitivas_host[i].triangulo.v1.y, primitivas_host[i].triangulo.v1.z, 0.0f));
            vertices_out.push_back(bvhvec4(primitivas_host[i].triangulo.v2.x, primitivas_host[i].triangulo.v2.y, primitivas_host[i].triangulo.v2.z, 0.0f));
            triangulos_host.push_back(primitivas_host[i]);
        }
    }

    const uint32_t num_triangulos = static_cast<uint32_t>(vertices_out.size() / 3);
    if (num_triangulos == 0) {
        num_primitivas_gpu = 0;
        return;
    }

    // Construir CWBVH
    BVH8_CWBVH* cwbvh = static_cast<BVH8_CWBVH*>(cwbvh_ptr);
    cwbvh->Build(vertices_out.data(), num_triangulos);

    // Si se vuelve a construir, liberar buffers anteriores.
    if (d_nodos) { cudaFree(d_nodos); d_nodos = nullptr; }
    if (d_tris) { cudaFree(d_tris); d_tris = nullptr; }
    if (primitivas_gpu) { cudaFree(primitivas_gpu); primitivas_gpu = nullptr; }

    // Calcular tamaños de memoria necesarios
    size_t bytes_nodos = cwbvh->usedBlocks * 5 * sizeof(bvhvec4); // Cada nodo CWBVH ocupa 5 bvhvec4 (80 bytes)
    size_t bytes_tris = cwbvh->bvh8.triCount * 4 * sizeof(bvhvec4); // Cada triángulo ocupa 4 bvhvec4 (64 bytes)

    // Reservar memoria en la GPU
    cudaMalloc(&d_nodos, bytes_nodos);
    cudaMalloc(&d_tris, bytes_tris);
    cudaMalloc(&primitivas_gpu, num_triangulos * sizeof(Primitiva));

    // Copiar nodos y tris a la GPU
    cudaMemcpy(d_nodos, cwbvh->bvh8Data, bytes_nodos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, cwbvh->bvh8Tris, bytes_tris, cudaMemcpyHostToDevice);

    // Copiar las primitivas ordenadas a la GPU
    cudaMemcpy(primitivas_gpu, triangulos_host.data(), 
               num_triangulos * sizeof(Primitiva), cudaMemcpyHostToDevice);

    // Actualizar punteros de la estructura
    datos_gpu.d_nodos = reinterpret_cast<float4*>(d_nodos);
    datos_gpu.d_tris = reinterpret_cast<float4*>(d_tris);
    datos_gpu.num_nodos = cwbvh->usedBlocks;
    datos_gpu.num_tris = cwbvh->bvh8.triCount;
    num_primitivas_gpu = num_triangulos;
}

void TinyBVH::obtenerInfo() {
    BVH8_CWBVH* cwbvh = static_cast<BVH8_CWBVH*>(cwbvh_ptr);
    printf("Número de nodos CWBVH (80 bytes c/u): %u\n", cwbvh->usedBlocks);
    printf("Número de primitivas referenciadas: %u\n", cwbvh->bvh8.triCount);
}

TinyBVHD_GPU TinyBVH::obtenerDatosGPU() { return datos_gpu; }

const TinyBVHD_GPU* TinyBVH::getNodosGPU() { return &datos_gpu; }

const Primitiva* TinyBVH::getPrimitivasGPU() { return primitivas_gpu; }

unsigned int TinyBVH::getNumNodos() { return static_cast<BVH8_CWBVH*>(cwbvh_ptr)->usedBlocks; }

unsigned int TinyBVH::getNumPrimitivas() { return num_primitivas_gpu; }