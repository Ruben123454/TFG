// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// bvh.cu
// ################

#include "bvh.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>

// Construcción del BVH
void ArbolBVH::construirBVH(std::vector<Primitiva>& primitivas_host) {
    if (primitivas_host.empty()) return;
    
    // Liberar memoria anterior
    liberar();
    
    num_primitivas = primitivas_host.size();
    
    // Construir BVH usando índices
    std::vector<NodoBVH> nodos_cpu;
    std::vector<int> indices(num_primitivas);
    for (int i = 0; i < num_primitivas; ++i) {
        indices[i] = i;
    }
    
    nodos_cpu.reserve(num_primitivas * 2);
    int nodo_actual = 0;
    
    int raiz_idx = construirRecursivo(primitivas_host, indices, 0, num_primitivas, nodo_actual, nodos_cpu);
    
    num_nodos = nodos_cpu.size();
    
    // Crear vector de primitivas ordenado según los índices del BVH
    std::vector<Primitiva> primitivas_ordenadas;
    primitivas_ordenadas.reserve(num_primitivas);
    for (int i = 0; i < num_primitivas; ++i) {
        primitivas_ordenadas.push_back(primitivas_host[indices[i]]);
    }
    
    // Copiar primitivas ordenadas a GPU
    cudaMalloc(&primitivas_gpu, num_primitivas * sizeof(Primitiva));
    cudaMemcpy(primitivas_gpu, primitivas_ordenadas.data(), 
               num_primitivas * sizeof(Primitiva), cudaMemcpyHostToDevice);
    
    // Copiar nodos a GPU
    cudaMalloc(&nodos_gpu, num_nodos * sizeof(NodoBVH));
    cudaMemcpy(nodos_gpu, nodos_cpu.data(), 
               num_nodos * sizeof(NodoBVH), cudaMemcpyHostToDevice);
}


// Construcción recursiva
int ArbolBVH::construirRecursivo(const std::vector<Primitiva>& primitivas,
                                            std::vector<int>& indices,
                                            int inicio, int fin, 
                                            int& nodo_actual,
                                            std::vector<NodoBVH>& nodos_cpu) {
    int idx_actual = nodo_actual++;
    
    // Asegurar espacio en el vector
    if (nodos_cpu.size() <= idx_actual) {
        nodos_cpu.resize(idx_actual + 1);
    }
    
    NodoBVH& nodo = nodos_cpu[idx_actual];
    
    // Calcular bounding box
    nodo.caja = calcularBoundingBox(primitivas, indices, inicio, fin);
    
    int numPrimitivas = fin - inicio;
    
    // Caso base: crear nodo hoja
    if (numPrimitivas <= 4) {
        nodo.izquierda = -1;
        nodo.derecha = -1;
        nodo.inicio = inicio;  // Índices en el array ordenado
        nodo.fin = fin;
        nodo.esHoja = true;
        return idx_actual;
    }
    
    int eje = nodo.caja.ejeMasLargo();
    std::sort(indices.begin() + inicio, indices.begin() + fin,
          [&primitivas, eje](int a, int b) {
              Vector3d centroA = primitivas[a].centroide();
              Vector3d centroB = primitivas[b].centroide();
              return centroA[eje] < centroB[eje];
          });
    int mid = inicio + numPrimitivas / 2;
    
    // Construir hijos recursivamente
    nodo.izquierda = construirRecursivo(primitivas, indices, inicio, mid, nodo_actual, nodos_cpu);
    nodo.derecha = construirRecursivo(primitivas, indices, mid, fin, nodo_actual, nodos_cpu);
    nodo.esHoja = false;
    
    return idx_actual;
}

// Calcular bounding box para un conjunto de primitivas
BoundingBox ArbolBVH::calcularBoundingBox(const std::vector<Primitiva>& primitivas,
                                                     const std::vector<int>& indices,
                                                     int inicio, int fin) {
    BoundingBox boundingBox;
    
    for (int i = inicio; i < fin; ++i) {
        int idx_prim = indices[i];
        BoundingBox primCaja = primitivas[idx_prim].obtenerCaja();
        boundingBox.expandir(primCaja);
    }
    
    return boundingBox;
}

// Liberar memoria GPU
void ArbolBVH::liberar() {
    if (nodos_gpu) {
        cudaFree(nodos_gpu);
        nodos_gpu = nullptr;
    }
    if (primitivas_gpu) {
        cudaFree(primitivas_gpu);
        primitivas_gpu = nullptr;
    }
    num_nodos = 0;
    num_primitivas = 0;
}

// Imprimir información del BVH
void ArbolBVH::obtenerInfo() const {
    printf("BVH: %d nodos, %d primitivas\n", num_nodos, num_primitivas);
}
