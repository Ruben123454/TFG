// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// cargarModelo.h
// ################

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "escenario.h"

using namespace std;

// Función para rotar un punto alrededor del eje Y
Vector3d rotarAlrededorEjeY(const Vector3d& punto, float anguloGrados) {
    float anguloRadianes = anguloGrados * M_PI / 180.0f;
    float cosAngulo = cos(anguloRadianes);
    float sinAngulo = sin(anguloRadianes);
    
    // Matriz de rotación alrededor del eje Y:
    // [ cosθ   0   sinθ ]
    // [  0     1    0   ]
    // [-sinθ   0   cosθ ]
    
    float x = punto.x * cosAngulo + punto.z * sinAngulo;
    float y = punto.y;
    float z = -punto.x * sinAngulo + punto.z * cosAngulo;
    
    return Vector3d(x, y, z);
}

// Función para rotar un punto alrededor del eje X
Vector3d rotarAlrededorEjeX(const Vector3d& punto, float anguloGrados) {
    float anguloRadianes = anguloGrados * M_PI / 180.0f;
    float cosAngulo = cos(anguloRadianes);
    float sinAngulo = sin(anguloRadianes);
    
    // Matriz de rotación alrededor del eje X:
    // [ 1     0        0    ]
    // [ 0   cosθ    -sinθ  ]
    // [ 0   sinθ     cosθ  ]
    
    float x = punto.x;
    float y = punto.y * cosAngulo - punto.z * sinAngulo;
    float z = punto.y * sinAngulo + punto.z * cosAngulo;
    
    return Vector3d(x, y, z);
}

// Función para rotar un punto alrededor del eje Z
Vector3d rotarAlrededorEjeZ(const Vector3d& punto, float anguloGrados) {
    float anguloRadianes = anguloGrados * M_PI / 180.0f;
    float cosAngulo = cos(anguloRadianes);
    float sinAngulo = sin(anguloRadianes);
    
    // Matriz de rotación alrededor del eje Z:
    // [ cosθ   -sinθ   0 ]
    // [ sinθ    cosθ   0 ]
    // [  0       0     1 ]
    
    float x = punto.x * cosAngulo - punto.y * sinAngulo;
    float y = punto.x * sinAngulo + punto.y * cosAngulo;
    float z = punto.z;
    
    return Vector3d(x, y, z);
}

Vector3d rotar(const Vector3d& punto, float anguloGradosX, float anguloGradosY, float anguloGradosZ) {
    Vector3d punto_rotado = rotarAlrededorEjeX(punto, anguloGradosX);
    punto_rotado = rotarAlrededorEjeY(punto_rotado, anguloGradosY);
    punto_rotado = rotarAlrededorEjeZ(punto_rotado, anguloGradosZ);
    return punto_rotado;
}

// Función para cargar un modelo 3D usando Assimp y convertirlo en primitivas
bool cargarModelo(const string& rutaModelo, vector<Primitiva>& host_prims, 
                    float escala, float rotacionGradosX, float rotacionGradosY, float rotacionGradosZ, const Vector3d& traslacion,
                    Color dif, Color spe, Color trans, Color emi, float ior) {
    cout << "\n=================================================" << endl;
    cout << "====        CARGA MALLA DE TRIÁNGULOS        ====" << endl;
    cout << "=================================================" << endl;
    
    Assimp::Importer importer;
    
    unsigned int flags = 
        aiProcess_Triangulate |
        aiProcess_GenNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_ImproveCacheLocality |
        aiProcess_ValidateDataStructure |
        aiProcess_PreTransformVertices;
    
    const aiScene* scene = importer.ReadFile(rutaModelo, flags);

    if (!scene || !scene->HasMeshes() || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) {
        cerr << "Error al cargar el modelo: " << rutaModelo << endl;
        cerr << importer.GetErrorString() << endl;
        return false;
    }
    
    // Contar triángulos totales para reservar espacio
    size_t total_triangles = 0;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];
        if (mesh->HasFaces()) {
            total_triangles += mesh->mNumFaces;
        }
    }
    host_prims.reserve(total_triangles);

    bool hay_normales_suaves = false;

    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh* mesh = scene->mMeshes[m];
        
        if (!mesh->HasFaces()) continue;

        cout << "Procesando malla " << m << ": " << mesh->mNumFaces << " caras, "
             << mesh->mNumVertices << " vértices, "
             << (mesh->HasNormals() ? "con normales" : "SIN NORMALES") << endl;


        // Recolectar triángulos
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) continue;

            const aiVector3D& a0 = mesh->mVertices[face.mIndices[0]];
            const aiVector3D& a1 = mesh->mVertices[face.mIndices[1]];
            const aiVector3D& a2 = mesh->mVertices[face.mIndices[2]];

            // Aplicar transformaciones en orden: ESCALA -> ROTACIÓN -> TRASLACIÓN
            Vector3d v0(a0.x, a0.y, a0.z);
            Vector3d v1(a1.x, a1.y, a1.z);
            Vector3d v2(a2.x, a2.y, a2.z);
            
            // Escalar
            Vector3d v0_scaled = v0 * escala;
            Vector3d v1_scaled = v1 * escala;
            Vector3d v2_scaled = v2 * escala;
            
            // Rotar
            Vector3d v0_rotated = rotar(v0_scaled, rotacionGradosX, rotacionGradosY, rotacionGradosZ);
            Vector3d v1_rotated = rotar(v1_scaled, rotacionGradosX, rotacionGradosY, rotacionGradosZ);
            Vector3d v2_rotated = rotar(v2_scaled, rotacionGradosX, rotacionGradosY, rotacionGradosZ);
            
            // Trasladar
            Vector3d v0_final = v0_rotated + traslacion;
            Vector3d v1_final = v1_rotated + traslacion;
            Vector3d v2_final = v2_rotated + traslacion;

            Primitiva prim = {};
            prim.tipo = TRIANGULO;
            prim.triangulo.v0 = v0_final;
            prim.triangulo.v1 = v1_final;
            prim.triangulo.v2 = v2_final;

            // Cargar normales de vértices
            if (mesh->HasNormals()) {
                Vector3d n0(mesh->mNormals[face.mIndices[0]].x,
                            mesh->mNormals[face.mIndices[0]].y,
                            mesh->mNormals[face.mIndices[0]].z);
                Vector3d n1(mesh->mNormals[face.mIndices[1]].x,
                            mesh->mNormals[face.mIndices[1]].y,
                            mesh->mNormals[face.mIndices[1]].z);
                Vector3d n2(mesh->mNormals[face.mIndices[2]].x,
                            mesh->mNormals[face.mIndices[2]].y,
                            mesh->mNormals[face.mIndices[2]].z);
                
                // Rotar normales
                prim.triangulo.v0_normal = rotar(n0, rotacionGradosX, rotacionGradosY, rotacionGradosZ).normalized();
                prim.triangulo.v1_normal = rotar(n1, rotacionGradosX, rotacionGradosY, rotacionGradosZ).normalized();
                prim.triangulo.v2_normal = rotar(n2, rotacionGradosX, rotacionGradosY, rotacionGradosZ).normalized();

                // Verificar si las normales son diferentes (suavizado)
                bool normales_diferentes = (prim.triangulo.v0_normal != prim.triangulo.v1_normal) ||
                                            (prim.triangulo.v0_normal != prim.triangulo.v2_normal);
                
                if (normales_diferentes) {
                    hay_normales_suaves = true;
                }

            } else {
                // Calcular normal geométrica del triángulo
                Vector3d edge1 = v1_final - v0_final;
                Vector3d edge2 = v2_final - v0_final;
                Vector3d normal_cara = edge1.cross(edge2).normalized();
                
                // Asignar la misma normal a los tres vértices
                prim.triangulo.v0_normal = normal_cara;
                prim.triangulo.v1_normal = normal_cara;
                prim.triangulo.v2_normal = normal_cara;
                
                cout << "  ADVERTENCIA: Malla sin normales, usando normales de cara" << endl;
            }

            prim.emision = emi;
            prim.difuso = dif;
            prim.especular = spe;
            prim.transmision = trans;
            prim.indice_refraccion = ior;

            host_prims.push_back(prim);
        }
    }

    if (host_prims.empty()) {
        cerr << "El modelo no contiene triángulos válidos: " << rutaModelo << endl;
        return false;
    }

    cout << "Modelo cargado exitosamente: " << rutaModelo 
         << ", Primitivas: " << host_prims.size()
         << ", Normales suaves: " << (hay_normales_suaves ? "SÍ" : "NO") << endl;

    return true;
}