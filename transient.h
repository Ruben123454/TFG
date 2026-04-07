// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// transient.h
// ################

#ifndef TRANSIENT_RENDER
#define TRANSIENT_RENDER

#include "imagen_gpu.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

class TransientRender {
public:
    Color* d_datos;
    int ancho, alto;
    int num_frames; // Número de frames temporales
    double t_start, t_end; // Rango de tiempo
    double sigma; // Desviación estándar del kernel gaussiano
    
    __host__ TransientRender(int w, int h, double start, double end, int n, double s = 1e-9) 
        : ancho(w), alto(h), t_start(start), t_end(end), num_frames(n), sigma(s), d_datos(nullptr) 
    {
        size_t total_pixels = (size_t)w * h * n;
        // Reservar memoria lineal en GPU para todos los frames
        // Usamos cudaMallocManaged para permitir usar RAM del sistema si falta VRAM
        cudaError_t err = cudaMallocManaged(&d_datos, total_pixels * sizeof(Color));
        if (err != cudaSuccess) {
            cerr << "Error reservando memoria Transient: " << cudaGetErrorString(err) << std::endl;
        } else {
            cudaMemset(d_datos, 0, total_pixels * sizeof(Color));
        }
    }

    __host__ void liberar() {
        if (d_datos) {
            cudaFree(d_datos);
            d_datos = nullptr;
        }
    }

    __host__ void limpiarAcumulado() {
        if (d_datos) {
            size_t total_pixels = (size_t)ancho * alto * num_frames;
            cudaMemset(d_datos, 0, total_pixels * sizeof(Color));
        }
    }

    /*
    // Método que agrega una muestra usando un kernel gaussiano
    __device__ void agregarMuestra(int x, int y, double tiempo, const Color& color) {
        if (x < 0 || x >= ancho || y < 0 || y >= alto) return;
        if (tiempo < t_start || tiempo > t_end) return;
        
        double duracion_frame = (t_end - t_start) / num_frames;

        // Fallback robusto: si sigma no es válido, acumulación en un único bin.
        if (sigma <= 0.0) {
            int frame_idx = static_cast<int>((tiempo - t_start) / duracion_frame);
            if (frame_idx < 0) frame_idx = 0;
            if (frame_idx >= num_frames) frame_idx = num_frames - 1;
            size_t idx = ((size_t)frame_idx * ancho * alto) + (y * ancho + x);
            d_datos[idx] = d_datos[idx] + color;
            return;
        }

        // Para reducir ruido negro: splat temporal normalizado (conservación de energía).
        int frame_centro = static_cast<int>((tiempo - t_start) / duracion_frame);
        int radio_frames = max(1, (int)ceil(3.0 * sigma / duracion_frame));
        int f0 = max(0, frame_centro - radio_frames);
        int f1 = min(num_frames - 1, frame_centro + radio_frames);

        double suma_pesos = 0.0;
        for (int frame_idx = f0; frame_idx <= f1; frame_idx++) {
            double tiempo_centro_frame = t_start + (frame_idx + 0.5) * duracion_frame;
            double dt = tiempo_centro_frame - tiempo;
            double peso = exp(-(dt * dt) / (2.0 * sigma * sigma));
            suma_pesos += peso;
        }

        if (suma_pesos <= 0.0) return;

        for (int frame_idx = f0; frame_idx <= f1; frame_idx++) {
            double tiempo_centro_frame = t_start + (frame_idx + 0.5) * duracion_frame;
            double dt = tiempo_centro_frame - tiempo;
            double peso = exp(-(dt * dt) / (2.0 * sigma * sigma));
            double peso_norm = peso / suma_pesos;

            size_t idx = ((size_t)frame_idx * ancho * alto) + (y * ancho + x);
            d_datos[idx] = d_datos[idx] + (color * (float)peso_norm);
        }
    }
    */
    
    /*
    // Método que agrega una muestra usando un kernel gaussiano
    __device__ void agregarMuestra(int x, int y, double tiempo, const Color& color) {
        if (x < 0 || x >= ancho || y < 0 || y >= alto) return;
        if (tiempo < t_start || tiempo > t_end) return;
        
        double duracion_frame = (t_end - t_start) / num_frames;
        
        double prefactor = 1.0 / (2.0 * M_PI * sigma * sigma);
        
        // Iterar sobre todos los frames
        for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
            // Calcular el tiempo central del frame actual
            double tiempo_centro_frame = t_start + (frame_idx + 0.5) * duracion_frame;

            // Calcular la diferencia de tiempo
            double diferencia_tiempo = tiempo_centro_frame - tiempo;
            
            // Kernel gaussiano
            double peso = prefactor * exp(-(diferencia_tiempo * diferencia_tiempo) / (2.0 * sigma * sigma));
            
            // Calcular el índice en la memoria lineal para el frame y píxel actual
            size_t idx = ((size_t)frame_idx * ancho * alto) + (y * ancho + x);
            d_datos[idx] = d_datos[idx] + (color * peso);
        }
    }
    */

    __device__ void agregarMuestra(int x, int y, float tiempo, const Color& color) {
        if (x < 0 || x >= ancho || y < 0 || y >= alto) return;
        if (tiempo < t_start || tiempo > t_end) return;
        
        float duracion_frame = (t_end - t_start) / num_frames;
        
        // Constante de normalización gaussiana
        float inv_sqrt_2pi_sigma = 1.0f / (sqrtf(2.0f * M_PI) * sigma);
        
        // Acotar el rango temporal a +- 3 sigmas para no iterar todo
        float radio_tiempo = 3.0f * sigma;
        float tiempo_min = tiempo - radio_tiempo;
        float tiempo_max = tiempo + radio_tiempo;
        
        // Calcular qué frames corresponden a esos tiempos
        int frame_min = max(0, (int)floor((tiempo_min - t_start) / duracion_frame));
        int frame_max = min(num_frames - 1, (int)floor((tiempo_max - t_start) / duracion_frame));

        // Calcular la suma de pesos para normalización
        float suma_pesos = 0.0f;
        for (int frame_idx = frame_min; frame_idx <= frame_max; frame_idx++) {
            float tiempo_centro_frame = t_start + (frame_idx + 0.5f) * duracion_frame;
            float diferencia_tiempo = tiempo_centro_frame - tiempo;
            float peso = inv_sqrt_2pi_sigma * expf(-(diferencia_tiempo * diferencia_tiempo) / (2.0f * sigma * sigma));
            suma_pesos += peso;
        }
        
        // Evitar división por cero
        if (suma_pesos < 1e-6f) return;

        // Depositar muestras normalizadas por la suma de pesos
        for (int frame_idx = frame_min; frame_idx <= frame_max; frame_idx++) {
            float tiempo_centro_frame = t_start + (frame_idx + 0.5f) * duracion_frame;
            float diferencia_tiempo = tiempo_centro_frame - tiempo;
            
            float peso = inv_sqrt_2pi_sigma * expf(-(diferencia_tiempo * diferencia_tiempo) / (2.0f * sigma * sigma));
            float peso_normalizado = peso / suma_pesos;
            
            size_t idx = ((size_t)frame_idx * ancho * alto) + (y * ancho + x);
            
            atomicAdd(&(d_datos[idx].r), color.r * peso_normalizado);
            atomicAdd(&(d_datos[idx].g), color.g * peso_normalizado);
            atomicAdd(&(d_datos[idx].b), color.b * peso_normalizado);
        }
    }

    /*
    // Método que agrega una muestra sin kernel (simple acumulación)
    __device__ void agregarMuestra(int x, int y, double tiempo, const Color& color) {
        if (x < 0 || x >= ancho || y < 0 || y >= alto) return;
        if (tiempo < t_start || tiempo > t_end) return;
        double duracion_frame = (t_end - t_start) / num_frames;
        int frame_idx = static_cast<int>((tiempo - t_start) / duracion_frame);
        if (frame_idx < 0 || frame_idx >= num_frames) return;
        size_t idx = ((size_t)frame_idx * ancho * alto) + (y * ancho + x);
        d_datos[idx] = d_datos[idx] + color;
    }
    */
   
    // Método para pasar un frame de la GPU a la CPU
    __host__ std::vector<Color> obtenerFrameHost(int frame_idx) {
        std::vector<Color> buffer(ancho * alto);
        if (d_datos && frame_idx >= 0 && frame_idx < num_frames) {
            size_t offset = (size_t)frame_idx * ancho * alto;
            // Copia de Device a Host
            cudaMemcpy(buffer.data(), d_datos + offset, ancho * alto * sizeof(Color), cudaMemcpyDeviceToHost);
        }
        return buffer;
    }

    // Método para debug: imprimir estadísticas de un frame
    __host__ void debugFrame(int frame_idx) {
        auto frame = obtenerFrameHost(frame_idx);
        double max_val = 0.0;
        double avg_val = 0.0;
        int pixels_iluminados = 0;
        
        for (int i = 0; i < ancho * alto; i++) {
            double lum = frame[i].r + frame[i].g + frame[i].b;
            if (lum > 0.001) pixels_iluminados++;
            if (lum > max_val) max_val = lum;
            avg_val += lum;
        }
        avg_val /= (ancho * alto);
        
        cout << "Frame " << frame_idx << ": max=" << max_val 
                  << ", avg=" << avg_val << ", pixels_iluminados=" << pixels_iluminados 
                  << "/" << (ancho * alto) << endl;
    }
};

#endif