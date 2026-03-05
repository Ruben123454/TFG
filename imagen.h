// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// imagen.h
// ################

#ifndef IMAGEN
#define IMAGEN

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include "color.h"

using namespace std;

class Imagen {
public:
    int anchura, altura;
    vector<Color> datos;

    Imagen(int anch, int alt) : anchura(anch), altura(alt), datos(anch*alt) {}

    // Métodos de acceso
    Color& at(int x, int y) { 
        return datos[y * anchura + x]; 
    }

    const Color& at(int x, int y) const {
        return datos[y * anchura + x];
    }

    void setPixel(int x, int y, const Color& color) {
        if (x >= 0 && x < anchura && y >= 0 && y < altura) {
            datos[y * anchura + x] = color;
        }
    }
    
    Imagen operator*(float scalar) const {
        Imagen resultado(anchura, altura);
        for (int y = 0; y < altura; ++y) {
            for (int x = 0; x < anchura; ++x) {
                Color p = at(x, y);
                resultado.at(x, y) = Color(p.r * scalar, p.g * scalar, p.b * scalar);
            }
        }
        return resultado;
    }

    // Métodos de post-procesado
    Imagen clamping() const {
        Imagen resultado(anchura, altura);
        for (int y = 0; y < altura; ++y) {
            for (int x = 0; x < anchura; ++x) {
                Color p = at(x, y);
                resultado.at(x, y) = Color(
                    min(max(p.r, 0.0f), 1.0f),
                    min(max(p.g, 0.0f), 1.0f),
                    min(max(p.b, 0.0f), 1.0f)
                );
            }
        }
        return resultado;
    }
    
    Imagen ecualizador() const {
        float max_val = 0.0;
        for (const auto& p : datos) {
            max_val = max(max_val, max(p.r, max(p.g, p.b)));
        }
        if (max_val > 1.0f) {
            Imagen resultado(anchura, altura);
            for (int y = 0; y < altura; ++y) {
                for (int x = 0; x < anchura; ++x) {
                    Color p = at(x, y);
                    resultado.at(x, y) = Color(p.r / max_val, p.g / max_val, p.b / max_val);
                }
            }
            return resultado;
        }
        return *this;
    }

    Imagen ecualizador_clamping(const float& max_val) const {
        Imagen resultado(anchura, altura);

        for (int y = 0; y < altura; ++y) {
            for (int x = 0; x < anchura; ++x) {
                Color p = at(x, y);

                Color norm(
                    p.r / max_val,
                    p.g / max_val,
                    p.b / max_val
                );

                resultado.at(x, y) = Color(
                    min(max(norm.r, 0.0f), 1.0f),
                    min(max(norm.g, 0.0f), 1.0f),
                    min(max(norm.b, 0.0f), 1.0f)
                );
            }
        }
        return resultado;
    }

    Imagen gamma() const {
        Imagen resultado = ecualizador();
        for (int y = 0; y < altura; ++y) {
            for (int x = 0; x < anchura; ++x) {
                Color p = resultado.at(x, y);

                auto safePow = [](float v) {
                    if (!std::isfinite(v) || v < 0.0f) return 0.0f;
                    v = clamp(v, 0.0f, 1.0f);
                    return std::pow(v, 1.0f / 2.2f);
                };

                resultado.at(x, y) = Color(safePow(p.r), safePow(p.g), safePow(p.b));
            }
        }
        return resultado;
    }

    Imagen clamp_gamma() const {
        Imagen resultado = gamma();
        return resultado.clamping();
    }

    Imagen reinhard() const {
        Imagen resultado(anchura, altura);
        
        const float delta = 0.0001f;
        const float a = 0.18f;
        
        // Precalcular la matriz de luminancias
        std::vector<float> luminancias(anchura * altura);
        
        // Primer paso: calcular y almacenar todas las luminancias
        for (int y = 0; y < altura; y++) {
            for (int x = 0; x < anchura; x++) {
                const Color& Color = at(x, y);
                // Usamos la fórmula estándar para luminancia Estándar ITU-R BT.709 (HDTV)
                luminancias[y * anchura + x] = 0.2126f * Color.r + 0.7152f * Color.g + 0.0722f * Color.b;
            }
        }
        
        // Calcular Lwa (luminancia promedio logarítmica)
        float sumaLogLuminancia = 0.0f;
        int totalColores = anchura * altura;
        
        for (int i = 0; i < totalColores; i++) {
            sumaLogLuminancia += log(delta + luminancias[i]);
        }
        float Lwa = exp(sumaLogLuminancia / totalColores);
        
        // Calcular Lmax (máxima luminancia)
        float Lmax = 0.0f;
        for (int i = 0; i < totalColores; i++) {
            if (luminancias[i] > Lmax) {
                Lmax = luminancias[i];
            }
        }
        float Lwhite = Lmax;
        
        // Aplicar el operador de Reinhard
        for (int y = 0; y < altura; y++) {
            for (int x = 0; x < anchura; x++) {
                const Color& ColorOriginal = at(x, y);
                Color& ColorResultado = resultado.at(x, y);
                
                float L = luminancias[y * anchura + x];
                float Lscaled = (a / Lwa) * L;
                float Ld = Lscaled * (1.0f + Lscaled / (Lwhite * Lwhite)) / (1.0f + Lscaled);
                
                if (L > 0.0f) {
                    float escala = Ld / L;
                    ColorResultado.r = ColorOriginal.r * escala;
                    ColorResultado.g = ColorOriginal.g * escala;
                    ColorResultado.b = ColorOriginal.b * escala;
                } else {
                    ColorResultado.r = 0.0f;
                    ColorResultado.g = 0.0f;
                    ColorResultado.b = 0.0f;
                }
            }
        }
        
        return resultado;
    }

    float acesFilmic(float x) const {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        
        x = x * 0.6f;  
        
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    }

    Imagen filmic() const {
        Imagen resultado(anchura, altura);
        for (int y = 0; y < altura; ++y) {
            for (int x = 0; x < anchura; ++x) {
                const Color& p = at(x, y);
                resultado.at(x, y) = Color(acesFilmic(p.r), acesFilmic(p.g), acesFilmic(p.b));
            }
        }
        return resultado;
    }
    
};

#endif