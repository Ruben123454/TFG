// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// bvh.h
// ################

#ifndef BVH
#define BVH

#include <vector>
#include "boundingBox.h"
#include "primitiva.h"

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

struct NodoBVH {
    BoundingBox caja;
    int izquierda;  // índice en el array, -1 si no existe
    int derecha;    // índice en el array, -1 si no existe  
    int inicio;     // índice primera primitiva
    int fin;        // índice última primitiva + 1
    bool esHoja;
    
    HOST_DEVICE NodoBVH() 
        : izquierda(-1), derecha(-1), inicio(-1), fin(-1), esHoja(false) {}
};

class ArbolBVH {
private:
    // Arrays planos para GPU
    NodoBVH* nodos_gpu;
    Primitiva* primitivas_gpu;
    int num_nodos;
    int num_primitivas;
    
    // Métodos de construcción
    int construirRecursivo(const std::vector<Primitiva>& primitivas,
                                            std::vector<int>& indices,
                                            int inicio, int fin, 
                                            int& nodo_actual,
                                            std::vector<NodoBVH>& nodos_cpu);
    
    // Calcular caja delimitadora para un conjunto de primitivas
    BoundingBox calcularBoundingBox(const std::vector<Primitiva>& primitivas,
                                                     const std::vector<int>& indices,
                                                     int inicio, int fin);

public:
    HOST_DEVICE ArbolBVH() 
        : nodos_gpu(nullptr), primitivas_gpu(nullptr), num_nodos(0), num_primitivas(0) {}
    
    ~ArbolBVH() { liberar(); }
    
    // Construir BVH en GPU
    void construirBVH(std::vector<Primitiva>& primitivas);
    void liberar();
    
    // Getters para uso en kernels CUDA
    HOST_DEVICE const NodoBVH* getNodosGPU() const { 
        return nodos_gpu; 
    }

    HOST_DEVICE const Primitiva* getPrimitivasGPU() const { 
        return primitivas_gpu;
    }

    HOST_DEVICE int getNumNodos() const { 
        return num_nodos; 
    }

    // Print info del BVH
    HOST_DEVICE void obtenerInfo() const;
};

#endif