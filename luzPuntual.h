// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// luzPuntual.h
// ################

#ifndef LUZ_PUNTUAL
#define LUZ_PUNTUAL

#include "color.h"
#include "vector3d.h"

struct LuzPuntual {
    Vector3d posicion;
    Color intensidad; // Color e intensidad de la luz
};

#endif