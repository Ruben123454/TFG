#include "RenderKernel.h"

#include "render.h"
#include "escenario.h"

#include <cfloat>

__device__ __forceinline__ float nrc_luminance(const Color& c) {
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

__global__ void kernelNrcDebugStats(
    const Color* __restrict__ buffer_pt,
    const Color* __restrict__ buffer_prediccion,
    const Color* __restrict__ buffer_throughput,
    int num_pixels,
    int step,
    NrcDebugStats* __restrict__ stats
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx = idx * step;
    if (pixelIdx >= num_pixels) return;

    Color pt = buffer_pt[pixelIdx];
    Color pred = buffer_prediccion[pixelIdx];
    Color th = buffer_throughput[pixelIdx];
    Color contrib = th * pred;

    bool pt_finite = isfinite(pt.r) && isfinite(pt.g) && isfinite(pt.b);
    bool pred_finite = isfinite(pred.r) && isfinite(pred.g) && isfinite(pred.b);
    bool th_finite = isfinite(th.r) && isfinite(th.g) && isfinite(th.b);
    bool contrib_finite = isfinite(contrib.r) && isfinite(contrib.g) && isfinite(contrib.b);

    atomicAdd((unsigned int*)&stats->sampled, 1u);

    // PT pre-composite
    if (!pt_finite) {
        atomicAdd((unsigned int*)&stats->pt_naninf, 1u);
    } else {
        atomicAdd(&stats->pt_sum_r, pt.r);
        atomicAdd(&stats->pt_sum_g, pt.g);
        atomicAdd(&stats->pt_sum_b, pt.b);
        atomicAdd(&stats->pt_sum_luma, nrc_luminance(pt));

        atomicMin(&stats->pt_min_r_bits, __float_as_uint(pt.r));
        atomicMin(&stats->pt_min_g_bits, __float_as_uint(pt.g));
        atomicMin(&stats->pt_min_b_bits, __float_as_uint(pt.b));
        atomicMax(&stats->pt_max_r_bits, __float_as_uint(pt.r));
        atomicMax(&stats->pt_max_g_bits, __float_as_uint(pt.g));
        atomicMax(&stats->pt_max_b_bits, __float_as_uint(pt.b));
        if (pt.r > 0.0f || pt.g > 0.0f || pt.b > 0.0f) {
            atomicAdd((unsigned int*)&stats->pt_nonzero, 1u);
        }
    }

    // NaN/Inf agregado para NRC (si cualquiera no es finito)
    if (!pred_finite || !th_finite || !contrib_finite) {
        atomicAdd((unsigned int*)&stats->naninf_any, 1u);
    }
    
    // Pred
    if (pred_finite) {
        atomicAdd(&stats->pred_sum_r, pred.r);
        atomicAdd(&stats->pred_sum_g, pred.g);
        atomicAdd(&stats->pred_sum_b, pred.b);
        atomicMin(&stats->pred_min_r_bits, __float_as_uint(pred.r));
        atomicMin(&stats->pred_min_g_bits, __float_as_uint(pred.g));
        atomicMin(&stats->pred_min_b_bits, __float_as_uint(pred.b));
        atomicMax(&stats->pred_max_r_bits, __float_as_uint(pred.r));
        atomicMax(&stats->pred_max_g_bits, __float_as_uint(pred.g));
        atomicMax(&stats->pred_max_b_bits, __float_as_uint(pred.b));
    }

    if (pred.r > 0.0f || pred.g > 0.0f || pred.b > 0.0f) {
        atomicAdd((unsigned int*)&stats->pred_nonzero, 1u);
    }

    // Throughput
    atomicAdd(&stats->th_sum_r, th.r);
    atomicAdd(&stats->th_sum_g, th.g);
    atomicAdd(&stats->th_sum_b, th.b);
    atomicMin(&stats->th_min_r_bits, __float_as_uint(th.r));
    atomicMin(&stats->th_min_g_bits, __float_as_uint(th.g));
    atomicMin(&stats->th_min_b_bits, __float_as_uint(th.b));
    atomicMax(&stats->th_max_r_bits, __float_as_uint(th.r));
    atomicMax(&stats->th_max_g_bits, __float_as_uint(th.g));
    atomicMax(&stats->th_max_b_bits, __float_as_uint(th.b));

    // Contrib
    if (contrib_finite) {
        atomicAdd(&stats->contrib_sum_r, contrib.r);
        atomicAdd(&stats->contrib_sum_g, contrib.g);
        atomicAdd(&stats->contrib_sum_b, contrib.b);
        atomicAdd(&stats->contrib_sum_luma, nrc_luminance(contrib));
        atomicMin(&stats->contrib_min_r_bits, __float_as_uint(contrib.r));
        atomicMin(&stats->contrib_min_g_bits, __float_as_uint(contrib.g));
        atomicMin(&stats->contrib_min_b_bits, __float_as_uint(contrib.b));
        atomicMax(&stats->contrib_max_r_bits, __float_as_uint(contrib.r));
        atomicMax(&stats->contrib_max_g_bits, __float_as_uint(contrib.g));
        atomicMax(&stats->contrib_max_b_bits, __float_as_uint(contrib.b));
    }

    if (contrib.r > 0.0f || contrib.g > 0.0f || contrib.b > 0.0f) {
        atomicAdd((unsigned int*)&stats->contrib_nonzero, 1u);
    }
}

__global__ void kernelRender(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer,
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red, bool modo_reconstruccion) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < ancho_img && y < alto_img) {
        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);

        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y,
                            imagen_directa, transientRenderer, frame_number,
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red, modo_reconstruccion);
    }
}

__global__ void kernelRender_tiny(const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer, 
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red, bool modo_reconstruccion) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < ancho_img && y < alto_img) {
        Escenario escena(const_cast<Primitiva*>(primitivas), num_primitivas, const_cast<LuzPuntual*>(luces), num_luces,
                        const_cast<Primitiva*>(primitivas_malla), num_primitivas_malla, nodos_bvh, primitivas_bvh, num_nodos_bvh);
        Render renderer(0.9, samples_per_pixel);

        renderer.renderizar(*camara, escena, ancho_img, alto_img, x, y,
                            imagen_directa, transientRenderer, frame_number,
                            dev_counter, buffer_registros, counter_train, max_cap_train,
                            buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red, modo_reconstruccion);
    }
}

__global__ void kernelComposite(Color* img_pt, Color* img_prediccion, Color* throughput_map,
                                int ancho, int alto, bool modo_reconstruccion) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * ancho + x;

    if(x < ancho && y < alto) {
        Color final = img_pt[idx];
        Color th = throughput_map[idx];
        Color indirecta_neuronal = img_prediccion[idx];

        if(!modo_reconstruccion){
            if (th.r > 0 || th.g > 0 || th.b > 0) {
                final = final + (th * indirecta_neuronal);
            }

            img_pt[idx] = final;
        } else {
            img_pt[idx] = (th * indirecta_neuronal);
        }
    }
}

__global__ void kernelTransientComposite(
    DatosMLP* buffer_inference, Color* buffer_prediccion, Color* buffer_throughput,
    TransientRender transientRenderer,
    int ancho, int alto,
    bool modo_reconstruccion
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * ancho + x;

    if (x < ancho && y < alto) {
        Color th = buffer_throughput[idx];
        if (th.r > 0 || th.g > 0 || th.b > 0) {
            Color contribucion = th * buffer_prediccion[idx];

            if (!isfinite(contribucion.r) || !isfinite(contribucion.g) || !isfinite(contribucion.b)) return;
            contribucion.r = fmaxf(0.0f, contribucion.r);
            contribucion.g = fmaxf(0.0f, contribucion.g);
            contribucion.b = fmaxf(0.0f, contribucion.b);

            double tiempo_total = (double)buffer_inference[idx].tiempo + (double)buffer_inference[idx].delta_t;
            transientRenderer.agregarMuestra(x, y, tiempo_total, contribucion);
        }
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
    d.tiempo = reg.head.tiempo;

    d.color.r = logf(1.0f + target_final.r);
    d.color.g = logf(1.0f + target_final.g);
    d.color.b = logf(1.0f + target_final.b);

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
    d.tiempo = 0.0f;

    if (reg.valido) {
        d.posicion = reg.tail.posicion;
        d.direccion = reg.tail.direccion;
        d.normal = reg.tail.normal;
        d.difuso = reg.tail.difuso;
        d.especular = reg.tail.especular;
        d.tiempo = reg.tail.tiempo;
    }

    dest_inference_inputs[idx] = d;
}

void launchKernelRender(dim3 gridSize, dim3 blockSize,
                        const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                        const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                        const NodoBVH* nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                        ImagenGPU imagen_directa,
                        TransientRender transientRenderer,
                        unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                        DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red, bool modo_reconstruccion) {
    kernelRender<<<gridSize, blockSize>>>(
        camara, primitivas, num_primitivas, luces, num_luces,
        primitivas_malla, num_primitivas_malla, ancho_img, alto_img, samples_per_pixel, frame_number,
        nodos_bvh, primitivas_bvh, num_nodos_bvh,
        imagen_directa,
        transientRenderer,
        dev_counter, buffer_registros, counter_train, max_cap_train,
        buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red, modo_reconstruccion
    );
}

void launchKernelRenderTiny(dim3 gridSize, dim3 blockSize,
                            const Camara* camara, const Primitiva* primitivas, int num_primitivas, const LuzPuntual* luces, int num_luces,
                            const Primitiva* primitivas_malla, int num_primitivas_malla, int ancho_img, int alto_img, int samples_per_pixel, int frame_number,
                            TinyBVHD_GPU nodos_bvh, const Primitiva* primitivas_bvh, int num_nodos_bvh,
                            ImagenGPU imagen_directa,
                            TransientRender transientRenderer, 
                            unsigned int* dev_counter, RegistroEntrenamiento* buffer_registros, unsigned int* counter_train, int max_cap_train,
                            DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red,
                            bool modo_reconstruccion) {
    kernelRender_tiny<<<gridSize, blockSize>>>(
        camara, primitivas, num_primitivas, luces, num_luces,
        primitivas_malla, num_primitivas_malla, ancho_img, alto_img, samples_per_pixel, frame_number,
        nodos_bvh, primitivas_bvh, num_nodos_bvh,
        imagen_directa,
        transientRenderer,
        dev_counter, buffer_registros, counter_train, max_cap_train,
        buffer_inference, buffer_throughput, usar_red_inferencia, entrenar_red, modo_reconstruccion
    );
}

void launchKernelComposite(dim3 gridSize, dim3 blockSize,
                           Color* img_pt, Color* img_prediccion, Color* throughput_map,
                           int ancho, int alto,
                           bool modo_reconstruccion) {
    kernelComposite<<<gridSize, blockSize>>>(img_pt, img_prediccion, throughput_map, ancho, alto, modo_reconstruccion);
}

void launchKernelTransientComposite(dim3 gridSize, dim3 blockSize,
                                    DatosMLP* buffer_inference, Color* buffer_prediccion, Color* buffer_throughput,
                                    TransientRender transientRenderer,
                                    int ancho, int alto,
                                    bool modo_reconstruccion) {
    kernelTransientComposite<<<gridSize, blockSize>>>(
        buffer_inference, buffer_prediccion, buffer_throughput, transientRenderer, ancho, alto, modo_reconstruccion
    );
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

void launchKernelNrcDebugStats(dim3 gridSize, dim3 blockSize,
    const Color* buffer_pt,
    const Color* buffer_prediccion,
    const Color* buffer_throughput,
    int num_pixels,
    int sample_count,
    NrcDebugStats* d_stats) {
    int safe_sample_count = sample_count;
    if (safe_sample_count < 1) {
        safe_sample_count = 1;
    }
    int step = num_pixels / safe_sample_count;
    if (step < 1) {
        step = 1;
    }
    kernelNrcDebugStats<<<gridSize, blockSize>>>(buffer_pt, buffer_prediccion, buffer_throughput, num_pixels, step, d_stats);
}
