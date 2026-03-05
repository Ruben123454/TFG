// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// imagen_utils.h
// ################

#ifndef IMAGEN_UTILS_H
#define IMAGEN_UTILS_H

#include <vector>
#include <string>
#include <png.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "imagen.h"

using namespace std;

// Función para guardar PNG desde un objeto Imagen
bool guardarPNG(const Imagen& imagen, const char* nombreFichero) {
    FILE* fp = fopen(nombreFichero, "wb");
    if (!fp) {
        cerr << "No se pudo abrir el archivo: " << nombreFichero << endl;
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        cerr << "No se pudo crear png_struct" << endl;
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        cerr << "No se pudo crear png_info" << endl;
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        cerr << "Error durante la escritura del PNG" << endl;
        return false;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, imagen.anchura, imagen.altura,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, info);

    std::vector<uint8_t> row(3 * imagen.anchura);

    for (int y = 0; y < imagen.altura; ++y) {
        for (int x = 0; x < imagen.anchura; ++x) {
            const Color& p = imagen.at(x, y);
            row[x*3 + 0] = static_cast<uint8_t>(round(clamp(p.r, 0.0f, 1.0f) * 255.0f));
            row[x*3 + 1] = static_cast<uint8_t>(round(clamp(p.g, 0.0f, 1.0f) * 255.0f));
            row[x*3 + 2] = static_cast<uint8_t>(round(clamp(p.b, 0.0f, 1.0f) * 255.0f));
        }
        png_write_row(png, row.data());
    }

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    
    return true;
}

#endif