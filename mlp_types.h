// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// mlp_types.h
// ################

#include "vector3d.h"
#include "color.h"

#ifndef MLP_TYPES
#define MLP_TYPES

struct DatosMLP {
    Vector3d posicion;  // Posición 3D del punto de intersección
    Vector3d direccion; // Dirección del rayo incidente
    Vector3d normal;    // Normal en el punto de intersección
    Color difuso;
    Color especular;
    

    // Output
    Color color;    // Color resultante del path tracing
};

// Estructura para definir los límites de la escena
struct SceneBounds {
    Vector3d min;
    Vector3d max;
};

// Datos que necesitamos para consultar a la red (Input)
struct DatosGeometricos {
    Vector3d posicion;
    Vector3d direccion;
    Vector3d normal;
    Color difuso;
    Color especular;
};

// Estructura completa para el buffer de entrenamiento
struct RegistroEntrenamiento {
    // HEAD: Donde queremos APRENDER (Input para train)
    DatosGeometricos head;
    
    // TAIL: Donde terminó el sufijo (Input para inferencia auxiliar)
    DatosGeometricos tail;
    
    // Datos del camino intermedio (Sufijo)
    Color luz_acumulada_sufijo;      // Radiance acumulada durante el sufijo
    Color throughput_sufijo;         // Throughput acumulado durante el sufijo
    Color factor_normalizacion;      // (Throughput_Head * Reflectancia_Head) para dividir al final
    
    bool valido; // Si el rayo encontró algo válido
};

#endif