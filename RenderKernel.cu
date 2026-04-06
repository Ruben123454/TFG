#include "RenderKernel.h"

#include "render.h"
#include "escenario.h"

__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < ancho_img && y < alto_img) {
        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);

        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y,
                            imagen_directa, frame_number,
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red);
    }
}

__global__ void kernelRender_tiny(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);

        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y,
                            imagen_directa, frame_number,
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red);
    
}

__global__ void kernelComposite(Color* img_pt, Color* img_prediccion, Color* throughput_map,
                                int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * ancho + x;

    if(x < ancho && y < alto) {
        Color final = img_pt[idx];
        Color th = throughput_map[idx];
        Color indirecta_neuronal = img_prediccion[idx];

        if (th.r > 0 || th.g > 0 || th.b > 0) {
            final = final + (th * indirecta_neuronal);
        }

        img_pt[idx] = final;
    }
}

__global__ void inicializarCamara(Camara* d_camara, int ancho_imagen, int alto_imagen) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float distancia_focal = 1.0f;
        float fov_grados = 45.0f;
        float fov_radianes = fov_grados * M_PI / 180.0f;
        float alto_plano = 2.0f * distancia_focal * tan(fov_radianes / 2.0f);
        float ancho_plano = alto_plano * (static_cast<float>(ancho_imagen) / alto_imagen);

        //Vector3d posicion(0, 0, 5);
        //Vector3d frente(0, 0, -1);
        //Vector3d arriba(0, 1, 0);

        Vector3d posicion(0, -0.75, 4.25);
        Vector3d frente(0, 0, -1);
        Vector3d arriba(0, 1, 0);

        //Vector3d posicion(2.6f, 0.2f, 4.25f);
        //Vector3d frente(-0.6f, -0.25f, -1.0f);
        //Vector3d arriba(0, 1, 0);

        *d_camara = Camara(posicion, frente, arriba, ancho_plano, alto_plano, distancia_focal);
    }
}

__global__ void kernelCalcularTargets(
    RegistroEntrenamiento* buffer_registros,
    Color* buffer_prediccion_tail,
    DatosMLP* buffer_training_final,
    int num_elementos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elementos) return;

    RegistroEntrenamiento reg = buffer_registros[idx];

    if (!reg.valido) {
        buffer_training_final[idx].color = Color(0,0,0);
        return;
    }

    Color iluminacion_tail = buffer_prediccion_tail[idx];
    Color radiance_total = reg.luz_acumulada_sufijo + (reg.throughput_sufijo * iluminacion_tail);
    Color denominador = reg.factor_normalizacion;

    float safe_eps = 1e-8f;

    Color target_final;
    target_final.r = radiance_total.r / fmaxf(denominador.r, safe_eps);
    target_final.g = radiance_total.g / fmaxf(denominador.g, safe_eps);
    target_final.b = radiance_total.b / fmaxf(denominador.b, safe_eps);

    float max_radiance = 100000.0f;
    target_final.r = fminf(target_final.r, max_radiance);
    target_final.g = fminf(target_final.g, max_radiance);
    target_final.b = fminf(target_final.b, max_radiance);

    if (isnan(target_final.r) || isinf(target_final.r)) target_final.r = 0.0f;
    if (isnan(target_final.g) || isinf(target_final.g)) target_final.g = 0.0f;
    if (isnan(target_final.b) || isinf(target_final.b)) target_final.b = 0.0f;

    DatosMLP d;
    d.posicion = reg.head.posicion;
    d.direccion = reg.head.direccion;
    d.normal = reg.head.normal;
    d.difuso = reg.head.difuso;
    d.especular = reg.head.especular;
    d.color = target_final;

    buffer_training_final[idx] = d;
}

__global__ void kernelPrepararInferenciaTail(
    const RegistroEntrenamiento* __restrict__ source_registros,
    DatosMLP* __restrict__ dest_inference_inputs,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const RegistroEntrenamiento& reg = source_registros[idx];

    DatosMLP d;
    d.posicion = Vector3d(0,0,0);
    d.direccion = Vector3d(0,0,0);
    d.normal = Vector3d(0,0,0);
    d.difuso = Color(0,0,0);
    d.especular = Color(0,0,0);
    d.color = Color(0,0,0);

    if (reg.valido) {
        d.posicion = reg.tail.posicion;
        d.direccion = reg.tail.direccion;
        d.normal = reg.tail.normal;
        d.difuso = reg.tail.difuso;
        d.especular = reg.tail.especular;
    }

    dest_inference_inputs[idx] = d;
}

void launchKernelRender(dim3 gridSize, dim3 blockSize,
                        const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                        const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                        const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                        ImagenGPU imagen_directa,
                        unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                        DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) {
    kernelRender<<<gridSize, blockSize>>>(
        camara, primitivas, num_primitivas, luces, num_luces,
        primitivas_malla, num_primitivas_malla, ancho_img, alto_img, samples_per_pixel, frame_number,
        nodos_bvh, primitivas_bvh, num_nodos_bvh,
        imagen_directa,
        dev_counter, buffer_registros, counter_train, max_cap_train,
        buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red
    );
}

void launchKernelRenderTiny(dim3 gridSize, dim3 blockSize,
                            const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red) {
    kernelRender_tiny<<<gridSize, blockSize>>>(
        camara, primitivas, num_primitivas, luces, num_luces,
        primitivas_malla, num_primitivas_malla, ancho_img, alto_img, samples_per_pixel, frame_number,
        nodos_bvh, primitivas_bvh, num_nodos_bvh,
        imagen_directa,
        dev_counter, buffer_registros, counter_train, max_cap_train,
        buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red
    );
}

void launchKernelComposite(dim3 gridSize, dim3 blockSize,
                           Color* img_pt, Color* img_prediccion, Color* throughput_map,
                           int ancho, int alto) {
    kernelComposite<<<gridSize, blockSize>>>(img_pt, img_prediccion, throughput_map, ancho, alto);
}

void launchInicializarCamara(Camara* d_camara, int ancho_imagen, int alto_imagen) {
    inicializarCamara<<<1, 1>>>(d_camara, ancho_imagen, alto_imagen);
}

void launchKernelCalcularTargets(dim3 gridSize, dim3 blockSize,
                                 RegistroEntrenamiento* buffer_registros,
                                 Color* buffer_prediccion_tail,
                                 DatosMLP* buffer_training_final,
                                 int num_elementos) {
    kernelCalcularTargets<<<gridSize, blockSize>>>(
        buffer_registros, buffer_prediccion_tail, buffer_training_final, num_elementos
    );
}

void launchKernelPrepararInferenciaTail(dim3 gridSize, dim3 blockSize,
                                        const RegistroEntrenamiento* source_registros,
                                        DatosMLP* dest_inference_inputs,
                                        int num_elements) {
    kernelPrepararInferenciaTail<<<gridSize, blockSize>>>(
        source_registros, dest_inference_inputs, num_elements
    );
}
