// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// color.h
// ################

#ifndef COLOR
#define COLOR

#include <cuda_runtime.h>

class Color {
public:
    float r, g, b;

    // Constructores para GPU
    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(float red, float green, float blue) : r(red), g(green), b(blue) {}

    // Operadores para GPU
    __host__ __device__ Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }

    __host__ __device__ Color operator+(const float& other) const {
        return Color(r + other, g + other, b + other);
    }

    __host__ __device__ Color operator-(const Color& other) const {
        return Color(r - other.r, g - other.g, b - other.b);
    }

    __host__ __device__ Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }

    __host__ __device__ Color operator*(const Color& other) const {
        return Color(r * other.r, g * other.g, b * other.b);
    }

    __host__ __device__ Color operator/(float scalar) const {
        // En GPU, si scalar es 0, devolvemos negro
        if (scalar == 0.0f) {
            return Color(0, 0, 0);
        }
        return Color(r / scalar, g / scalar, b / scalar);
    }

    __host__ __device__ Color operator/(const Color& other) const {
        // En GPU, si algún componente de other es 0, devolvemos negro
        float red = (other.r == 0.0f) ? 0.0f : r / other.r;
        float green = (other.g == 0.0f) ? 0.0f : g / other.g;
        float blue = (other.b == 0.0f) ? 0.0f : b / other.b;
        return Color(red, green, blue);
    }

    __host__ __device__ float max() const {
        // Devolver el componente máximo entre r, g y b
        float m;
        if(r >= g){
            m = r;
        }else{
            m = g;
        }
        if(b > m){
            m = b;
        }
        return m;
    }
};

#endif