#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "imagen.h"
#include "imagen_utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

struct MetricsResult {
    float mse = 0.0f;
    float mrse = 0.0f;
    float psnr = 0.0f;
    float ssim = 0.0f;
};

inline std::string toLowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline std::string getFileExtensionLower(const std::string& path) {
    return toLowerCopy(std::filesystem::path(path).extension().string());
}

inline bool readColorImageFromBinaryFile(const std::string& path, int width, int height, std::vector<Color>& out, std::string& error, int frameIndex = 0) {
    if (width <= 0 || height <= 0) {
        error = "Se necesita un ancho y alto validos para leer .bin/.raw.";
        return false;
    }
    if (frameIndex < 0) {
        error = "El indice de frame debe ser mayor o igual que 0.";
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "No se pudo abrir el archivo: " + path;
        return false;
    }

    const size_t num_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    out.resize(num_pixels);

    const std::streamoff frameOffset = static_cast<std::streamoff>(frameIndex) * static_cast<std::streamoff>(num_pixels) * static_cast<std::streamoff>(3 * sizeof(float));
    file.seekg(frameOffset, std::ios::beg);
    if (!file) {
        error = "No se pudo posicionar en el frame " + std::to_string(frameIndex) + " del binario: " + path;
        return false;
    }

    for (size_t i = 0; i < num_pixels; ++i) {
        float r = 0.0f, g = 0.0f, b = 0.0f;
        file.read(reinterpret_cast<char*>(&r), sizeof(float));
        file.read(reinterpret_cast<char*>(&g), sizeof(float));
        file.read(reinterpret_cast<char*>(&b), sizeof(float));
        if (!file) {
            error = "El archivo binario no contiene suficientes datos para " + std::to_string(width) + "x" + std::to_string(height) + " pixeles.";
            return false;
        }
        out[i] = Color(r, g, b);
    }

    return true;
}

inline bool loadImageAsColorBuffer(const std::string& path, int fallbackWidth, int fallbackHeight, std::vector<Color>& out, int& width, int& height, std::string& error) {
    const std::string ext = getFileExtensionLower(path);

    if (ext == ".png") {
        int w = 0, h = 0;
        std::vector<uint8_t> png_data = cargarPNG(path, w, h);
        if (png_data.empty() || w <= 0 || h <= 0) {
            error = "No se pudo cargar el PNG: " + path;
            return false;
        }

        width = w;
        height = h;
        out = convertirPNGaColor(png_data, width, height);
        return true;
    }

    width = fallbackWidth;
    height = fallbackHeight;
    if (width <= 0 || height <= 0) {
        error = "Para archivos .bin/.raw debes indicar un ancho y alto validos.";
        return false;
    }

    if (ext == ".bin" || ext == ".raw" || ext.empty()) {
        return readColorImageFromBinaryFile(path, width, height, out, error);
    }

    error = "Formato no soportado: " + ext + ". Usa .png, .bin o .raw.";
    return false;
}

inline std::vector<uint8_t> convertirColorARGB8(const std::vector<Color>& colors) {
    std::vector<uint8_t> rgba(colors.size() * 4);
    for (size_t i = 0; i < colors.size(); ++i) {
        const Color& c = colors[i];
        rgba[i * 4 + 0] = static_cast<uint8_t>(std::round(std::clamp(c.r, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 1] = static_cast<uint8_t>(std::round(std::clamp(c.g, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 2] = static_cast<uint8_t>(std::round(std::clamp(c.b, 0.0f, 1.0f) * 255.0f));
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

inline float linearToSrgb(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    if (value <= 0.0031308f) {
        return 12.92f * value;
    }
    return 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

inline std::vector<uint8_t> convertirColorARGB8(const std::vector<Color>& colors, bool aplicarGamma) {
    std::vector<uint8_t> rgba(colors.size() * 4);
    for (size_t i = 0; i < colors.size(); ++i) {
        const Color& c = colors[i];
        const float r = aplicarGamma ? linearToSrgb(c.r) : std::clamp(c.r, 0.0f, 1.0f);
        const float g = aplicarGamma ? linearToSrgb(c.g) : std::clamp(c.g, 0.0f, 1.0f);
        const float b = aplicarGamma ? linearToSrgb(c.b) : std::clamp(c.b, 0.0f, 1.0f);
        rgba[i * 4 + 0] = static_cast<uint8_t>(std::round(r * 255.0f));
        rgba[i * 4 + 1] = static_cast<uint8_t>(std::round(g * 255.0f));
        rgba[i * 4 + 2] = static_cast<uint8_t>(std::round(b * 255.0f));
        rgba[i * 4 + 3] = 255;
    }
    return rgba;
}

float calcularMSE(const uint8_t* d_ref, const uint8_t* d_pred, int num_pixels);
float calcularMRSE(const Color* d_img, const Color* d_gt, int num_pixels);
float calcularPSNR(const uint8_t* d_ref, const uint8_t* d_pred, int num_pixels);
float calcularSSIM(const uint8_t* d_ref, const uint8_t* d_pred, int width, int height);
inline MetricsResult calcularMetricasDesdeArchivos(
    const std::string& groundtruthPath,
    const std::string& imagePath,
    int fallbackWidth,
    int fallbackHeight,
    bool calcMSE,
    bool calcMRSE,
    bool calcPSNR,
    bool calcSSIM,
    std::string& errorMessage)
{
    errorMessage.clear();

    if (!calcMSE && !calcMRSE && !calcPSNR && !calcSSIM) {
        errorMessage = "Selecciona al menos una metrica.";
        return {};
    }

    std::vector<Color> gt_color;
    std::vector<Color> img_color;
    int gt_width = 0, gt_height = 0;
    int img_width = 0, img_height = 0;
    const std::string gt_ext = getFileExtensionLower(groundtruthPath);
    const std::string img_ext = getFileExtensionLower(imagePath);

    if (!loadImageAsColorBuffer(groundtruthPath, fallbackWidth, fallbackHeight, gt_color, gt_width, gt_height, errorMessage)) {
        return {};
    }
    if (!loadImageAsColorBuffer(imagePath, fallbackWidth, fallbackHeight, img_color, img_width, img_height, errorMessage)) {
        return {};
    }

    if (gt_width != img_width || gt_height != img_height) {
        errorMessage = "Las imagenes no tienen la misma resolucion: GT=" + std::to_string(gt_width) + "x" + std::to_string(gt_height) +
                       " vs IMG=" + std::to_string(img_width) + "x" + std::to_string(img_height);
        return {};
    }

    const int num_pixels = gt_width * gt_height;
    MetricsResult result;

    // MSE usa espacio LDR (sRGB uint8)
    if (calcMSE) {
        const bool gt_aplicar_gamma = (gt_ext != ".png");
        const bool img_aplicar_gamma = (img_ext != ".png");

        const std::vector<uint8_t> gt_rgba = convertirColorARGB8(gt_color, gt_aplicar_gamma);
        const std::vector<uint8_t> img_rgba = convertirColorARGB8(img_color, img_aplicar_gamma);

        uint8_t* d_gt_rgba = nullptr;
        uint8_t* d_img_rgba = nullptr;
        cudaMalloc(&d_gt_rgba, gt_rgba.size() * sizeof(uint8_t));
        cudaMalloc(&d_img_rgba, img_rgba.size() * sizeof(uint8_t));
        cudaMemcpy(d_gt_rgba, gt_rgba.data(), gt_rgba.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_img_rgba, img_rgba.data(), img_rgba.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

        result.mse = calcularMSE(d_gt_rgba, d_img_rgba, num_pixels);

        cudaFree(d_gt_rgba);
        cudaFree(d_img_rgba);
    }

    // MRSE usa espacio HDR (float lineal)
    if (calcMRSE) {
        Color* d_gt = nullptr;
        Color* d_img = nullptr;
        cudaMalloc(&d_gt, num_pixels * sizeof(Color));
        cudaMalloc(&d_img, num_pixels * sizeof(Color));
        cudaMemcpy(d_gt, gt_color.data(), num_pixels * sizeof(Color), cudaMemcpyHostToDevice);
        cudaMemcpy(d_img, img_color.data(), num_pixels * sizeof(Color), cudaMemcpyHostToDevice);

        result.mrse = calcularMRSE(d_img, d_gt, num_pixels);

        cudaFree(d_gt);
        cudaFree(d_img);
    }

    // PSNR y SSIM usan espacio LDR (sRGB uint8)
    if (calcPSNR || calcSSIM) {
        const bool gt_aplicar_gamma = (gt_ext != ".png");
        const bool img_aplicar_gamma = (img_ext != ".png");

        const std::vector<uint8_t> gt_rgba = convertirColorARGB8(gt_color, gt_aplicar_gamma);
        const std::vector<uint8_t> img_rgba = convertirColorARGB8(img_color, img_aplicar_gamma);

        uint8_t* d_gt_rgba = nullptr;
        uint8_t* d_img_rgba = nullptr;
        cudaMalloc(&d_gt_rgba, gt_rgba.size() * sizeof(uint8_t));
        cudaMalloc(&d_img_rgba, img_rgba.size() * sizeof(uint8_t));
        cudaMemcpy(d_gt_rgba, gt_rgba.data(), gt_rgba.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_img_rgba, img_rgba.data(), img_rgba.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

        if (calcPSNR) {
            result.psnr = calcularPSNR(d_gt_rgba, d_img_rgba, num_pixels);
        }
        if (calcSSIM) {
            result.ssim = calcularSSIM(d_gt_rgba, d_img_rgba, gt_width, gt_height);
        }

        cudaFree(d_gt_rgba);
        cudaFree(d_img_rgba);
    }

    return result;
}

inline void imprimirResultadosMetricas(const MetricsResult& metrics, bool calcMSE, bool calcMRSE, bool calcPSNR, bool calcSSIM) {
    cout << "\n==========================================" << endl;
    cout << "====         RESULTADOS METRICAS      ====" << endl;
    cout << "==========================================" << endl;
    if (calcMSE) cout << "MSE  : " << metrics.mse << endl;
    if (calcMRSE) cout << "MRSE : " << metrics.mrse << endl;
    if (calcPSNR) cout << "PSNR : " << metrics.psnr << " dB" << endl;
    if (calcSSIM) cout << "SSIM : " << metrics.ssim << endl;
}
