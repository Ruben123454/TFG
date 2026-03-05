// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// vector3d.h
// ################

#ifndef VECTOR3D
#define VECTOR3D

#include <cuda_runtime.h>
#include <cmath>

class Vector3d {
public:
    double x, y, z;

    // Constructores
    __device__ __host__ Vector3d() : x(0), y(0), z(0) {}
    __device__ __host__ Vector3d(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    // Operadores aritméticos
    __host__ __device__ Vector3d operator+(const Vector3d& other) const {
        return Vector3d(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vector3d operator-(const Vector3d& other) const {
        return Vector3d(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ Vector3d operator*(double scalar) const {
        return Vector3d(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ Vector3d operator/(double scalar) const {
        return Vector3d(x / scalar, y / scalar, z / scalar);
    }

    __host__ __device__ Vector3d operator-() const {
        return Vector3d(-x, -y, -z);
    }

    __host__ __device__ Vector3d& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // Producto escalar
    __host__ __device__ double dot(const Vector3d& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    // Producto vectorial
    __host__ __device__ Vector3d cross(const Vector3d& other) const {
        return Vector3d(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Normalización
    __host__ __device__ Vector3d normalized() const {
        double len = length();
        if (len > 0.0) {
            return *this * (1.0 / len);
        }
        return *this;
    }

    __host__ __device__ void normalize() {
        double len = length();
        if (len > 0.0) {
            x /= len;
            y /= len;
            z /= len;
        }
    }

    // Longitud
    __host__ __device__ double length() const {
        return sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ double lengthSquared() const {
        return x * x + y * y + z * z;
    }

    // Distancia entre puntos
    __host__ __device__ double distance(const Vector3d& other) const {
        return (*this - other).length();
    }

    // Operadores de comparación
    __host__ __device__ bool operator==(const Vector3d& other) const {
        return (x == other.x && y == other.y && z == other.z);
    }

    __host__ __device__ bool operator!=(const Vector3d& other) const {
        return !(*this == other);
    }

    // Producto por componentes (Hadamard product)
    __host__ __device__ Vector3d operator*(const Vector3d& other) const {
        return Vector3d(x * other.x, y * other.y, z * other.z);
    }

    // División por componentes
    __host__ __device__ Vector3d operator/(const Vector3d& other) const {
        return Vector3d(x / other.x, y / other.y, z / other.z);
    }

    // Norm
    __host__ __device__ double norm() const {
        return sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ double operator[](int index) const {
        if (index == 0) return x;
        else if (index == 1) return y;
        else return z;
    }

    // cwiseMin y cwiseMax
    __host__ __device__ Vector3d cwiseMin(const Vector3d& other) const {
        return Vector3d(fmin(x, other.x), fmin(y, other.y), fmin(z, other.z));
    }
    __host__ __device__ Vector3d cwiseMax(const Vector3d& other) const {
        return Vector3d(fmax(x, other.x), fmax(y, other.y), fmax(z, other.z));
    }
};

// Operadores globales
__host__ __device__ inline Vector3d operator*(double scalar, const Vector3d& vec) {
    return vec * scalar;
}

__host__ __device__ inline Vector3d operator/(double scalar, const Vector3d& vec) {
    return Vector3d(scalar / vec.x, scalar / vec.y, scalar / vec.z);
}

#endif