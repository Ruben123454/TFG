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
inline bool guardarPNG(const Imagen& imagen, const char* nombreFichero) {
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

// Cargar imagen PNG desde archivo
inline std::vector<uint8_t> cargarPNG(const std::string& filename, int& width, int& height) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "No se pudo abrir PNG: " << filename << std::endl;
        width = height = 0;
        return {};
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fclose(fp);
        return {};
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        return {};
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return {};
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    std::vector<uint8_t> image_data(width * height * 4);
    std::vector<png_bytep> row_pointers(height);
    
    for (int y = 0; y < height; y++) {
        row_pointers[y] = image_data.data() + y * width * 4;
    }

    png_read_image(png, row_pointers.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);

    return image_data;
}

// Convertir PNG (uint8) a Color
inline std::vector<Color> convertirPNGaColor(const std::vector<uint8_t>& png_data, int width, int height) {
    std::vector<Color> result(width * height);
    for (int i = 0; i < width * height; i++) {
        result[i].r = (float)png_data[i * 4 + 0] / 255.0f;
        result[i].g = (float)png_data[i * 4 + 1] / 255.0f;
        result[i].b = (float)png_data[i * 4 + 2] / 255.0f;
    }
    return result;
}

#endif