// ################
// Autores: 
// Mir Ramos, Rubén 869039
//
// render.h
// ################

#ifndef RENDER
#define RENDER

#include <cuda_runtime.h>
#include "camara.h"
#include "escenario.h"
#include "imagen_gpu.h"
#include "color.h"
#include "primitiva.h"
#include "mlp_types.h"
#include "rng.h"

const float pi = 3.14159265358979323846f;

class Render {
public:
    double russian_roulette_prob;
    int samples_per_pixel;

    const float EPSILON = 1e-3f;

    // Constructor para GPU
    __device__ Render(double rr = 0.9, int spp = 100) : 
                            russian_roulette_prob(rr), samples_per_pixel(spp) {}

    __device__ Color lanzarRayoIterativo(const Rayo& r_inicial, const Escenario& escena, uint32_t& rng_seed, int px, int py, int sample_idx, TransientRender& transientRenderer, 
                                    RegistroEntrenamiento& registro_train, bool& guardar_train,
                                    DatosMLP& datos_inf, Color& throughput_inf, bool& necesita_inf, bool es_ruta_entrenamiento, bool usar_red_inferencia,
                                    bool modo_reconstruccion) {
        Color color(0, 0, 0);
        Color camino(1, 1, 1); 
        Rayo rayo_actual = r_inicial;
        double current_ior = 1.0;
        double tiempo_acumulado = 0.0;

        guardar_train = false;
        necesita_inf = false;

        // Variables para capturar el target de entrenamiento
        Color throughput_at_train_vertex(0,0,0);
        Color color_acumulado_pre_train(0,0,0);
        bool punto_entrenamiento_encontrado = false;
        int rebotes_sufijo = 0;

        // Variables footprint
        float a_0 = 0.0f; // Footprint proyectado del píxel primario
        float a_current = 0.0f; // Footprint acumulado del subpath hasta el rebote actual
        float a_suffix = 0.0f; // Footprint exclusivo para el sufijo de entrenamiento
        float a_prefix = 0.0f; // Footprint del prefijo al momento de grabar el head (para terminar el sufijo)
        bool footprint_suficiente = false;
        
        int profundidad = 0;
        
        float pdf_ultimo_rebote = 1.0f;
        bool ultimo_fue_especular = false;

        int max_depth_actual = 20;

        bool es_ruta_insesgada = es_ruta_entrenamiento && (pcg32_float(rng_seed) < 0.0625f);

        double tiempo_acumulado_NEE = 0.0; // Tiempo acumulado solo para NEE (iluminación directa)
        double tiempo_camara_head = 0.0;

        while(profundidad < max_depth_actual) {

            float t_interseccion;
            const Primitiva* primitiva_intersectada;
            float u = 0.0f, v = 0.0f;
            
            if(escena.intersectaPrimitivasBVH_tiny(rayo_actual, t_interseccion, &primitiva_intersectada, u, v)) {

                Vector3d punto_interseccion = rayo_actual.origen() + rayo_actual.direccion() * t_interseccion;
                Vector3d normal = escena.calcularNormal(primitiva_intersectada, punto_interseccion, u, v);

                Vector3d direccion_vista = -rayo_actual.direccion().normalized();
                
                float cos_theta = abs(normal.dot(-rayo_actual.direccion()));

                bool es_difuso = (primitiva_intersectada->difuso.max() > EPSILON);
                bool es_especular = (primitiva_intersectada->especular.max() > EPSILON);          

                tiempo_acumulado += (t_interseccion * current_ior) / 299792458.0;

                float dist_sq = t_interseccion * t_interseccion;
                if (profundidad == 0) {
                    // a_0: Footprint proyectado del píxel primario
                    a_0 = dist_sq / (4.0f * pi * fmaxf(cos_theta, 1e-4f));
                    a_current = 0.0f; // El footprint del subpath empieza en 0
                } else {
                    // a(x1...xn): Acumulamos la dispersión del subpath
                    float spread_step = dist_sq / (fmaxf(pdf_ultimo_rebote, 1e-4f) * fmaxf(cos_theta, 1e-4f));
                    
                    if (!punto_entrenamiento_encontrado) {
                        a_current += spread_step;
                    } else {
                        a_suffix += spread_step;
                    }
                }

                footprint_suficiente = (a_current >= a_0 * 0.01f);

                Color Le = primitiva_intersectada->emision;
                if (Le.max() > 0.0f) {
                    Color contrib = camino * Le;
                    color = color + contrib;

                    if (punto_entrenamiento_encontrado) {
                        registro_train.tail.posicion = punto_interseccion;
                        registro_train.tail.direccion = -rayo_actual.direccion().normalized();
                        registro_train.tail.normal = normal;
                        registro_train.tail.difuso = Color(0,0,0);
                        registro_train.tail.especular = Color(0,0,0);


                        registro_train.tail.tiempo = 0.0f;
                        // Tiempo desde el head hasta la luz (target para el MLP)
                        double delta_t_head_tail = tiempo_acumulado - tiempo_camara_head;
                        registro_train.head.tiempo = (float)delta_t_head_tail;


                        // La luz total recolectada por el sufijo
                        registro_train.luz_acumulada_sufijo = color - color_acumulado_pre_train;
                        registro_train.throughput_sufijo = Color(0,0,0);
                        registro_train.factor_normalizacion = throughput_at_train_vertex;
                        registro_train.valido = true;
                        guardar_train = true;
                    }

                    // Solo depositar en transient si no estamos en el sufijo de una ruta de entrenamiento
                    if (!(es_ruta_entrenamiento && punto_entrenamiento_encontrado)) {
                        transientRenderer.agregarMuestra(px, py, tiempo_acumulado, contrib);
                    }
                    break;
                }

                // NEE (Luz Directa)
                if(es_difuso && !modo_reconstruccion) {
                    Vector3d punto_sombra = punto_interseccion + normal * EPSILON;
                    bool depositar_transient_nee = !(es_ruta_entrenamiento && punto_entrenamiento_encontrado);
                    Color L_dir = escena.calcularLuzDirecta(punto_interseccion, normal, primitiva_intersectada, 
                                                                transientRenderer, px, py, tiempo_acumulado, 
                                                                tiempo_acumulado_NEE, camino, current_ior, depositar_transient_nee);
                    Color contrib_nee = camino * L_dir;
                    color = color + contrib_nee;
                }
                

                if (punto_entrenamiento_encontrado) {
                    rebotes_sufijo++;

                    bool terminar_sufijo = !es_ruta_insesgada && (a_suffix >= a_prefix);
                    Color reflectancia_tail = primitiva_intersectada->difuso + primitiva_intersectada->especular + primitiva_intersectada->transmision;

                    if (terminar_sufijo) { // Bootstrapping del sufijo de entrenamiento

                        // Pasar de [-1, 1] a [0, 1]
                        direccion_vista = direccion_vista.normalized();
                        normal = normal.normalized();
                        Vector3d dir_mapped = Vector3d(direccion_vista.x * 0.5f + 0.5f, 
                                                    direccion_vista.y * 0.5f + 0.5f, 
                                                    direccion_vista.z * 0.5f + 0.5f);
                        Vector3d norm_mapped = Vector3d(normal.x * 0.5f + 0.5f, 
                                                        normal.y * 0.5f + 0.5f, 
                                                        normal.z * 0.5f + 0.5f);

                        // Tiempo desde la luz hasta el tail
                        double tiempo_luz_tail = fmax(0.0, tiempo_acumulado_NEE - tiempo_acumulado);
                        registro_train.tail.tiempo = (float)tiempo_luz_tail;
                        // Tiempo desde el head hasta el tail
                        double delta_t_head_tail = tiempo_acumulado - tiempo_camara_head;
                        // Tiempo desde el head hasta la luz (target para el MLP)
                        registro_train.head.tiempo = (float)(tiempo_luz_tail + delta_t_head_tail);

                        registro_train.tail.posicion = punto_interseccion;
                        registro_train.tail.direccion = dir_mapped;
                        registro_train.tail.normal = norm_mapped;
                        registro_train.tail.difuso = primitiva_intersectada->difuso;
                        registro_train.tail.especular = primitiva_intersectada->especular;
                        registro_train.luz_acumulada_sufijo = color - color_acumulado_pre_train;
                        
                        registro_train.throughput_sufijo = camino * reflectancia_tail;
                        registro_train.factor_normalizacion = throughput_at_train_vertex;
                        registro_train.valido = true;
                        guardar_train = true;
                        break;
                    }
                }

                // Decidir si el rebote actual es parte del sufijo de entrenamiento o no, y capturar datos para la inferencia
                if((es_difuso && !es_especular && footprint_suficiente) || modo_reconstruccion) {
                    Color reflectancia = primitiva_intersectada->difuso + primitiva_intersectada->especular + primitiva_intersectada->transmision;
                    float min_refl = 1e-3f; 
                    reflectancia.r = fmaxf(reflectancia.r, min_refl);
                    reflectancia.g = fmaxf(reflectancia.g, min_refl);
                    reflectancia.b = fmaxf(reflectancia.b, min_refl);

                    // Pasar de [-1, 1] a [0, 1]
                    Vector3d dir_mapped = Vector3d(direccion_vista.x * 0.5f + 0.5f, 
                                                direccion_vista.y * 0.5f + 0.5f, 
                                                direccion_vista.z * 0.5f + 0.5f);
                    
                    Vector3d norm_mapped = Vector3d(normal.x * 0.5f + 0.5f, 
                                                    normal.y * 0.5f + 0.5f, 
                                                    normal.z * 0.5f + 0.5f);
                                                    
                    if (!es_ruta_entrenamiento && usar_red_inferencia) { // Capturar datos para inferencia
                        // Muestrear T_target con jittering temporal estratificado
                        double t_start = transientRenderer.t_start; 
                        double t_final = transientRenderer.t_end;
                        int num_bins_t = transientRenderer.num_frames;
                        double dt_bin = (t_final - t_start) / (double)num_bins_t;
                        unsigned int pixel_hash = (unsigned int)(px * 73856093u) ^ (unsigned int)(py * 19349663u);
                        int bin_idx = (sample_idx + (int)(pixel_hash % (unsigned int)num_bins_t)) % num_bins_t;
                        float jitter = pcg32_float(rng_seed); // jitter intra-bin en [0,1)
                        double T_target = t_start + ((double)bin_idx + (double)jitter) * dt_bin;

                        // Evitar salir por redondeo en el último bin
                        if (T_target >= t_final) {
                            T_target = t_final - 1e-12;
                        }

                        // Tiempo que le pasamos a la red quitando el tiempo acumulado del camino hasta ahora, 
                        // para que la red aprenda a predecir el tiempo restante hasta la luz
                        double t_input = T_target - tiempo_acumulado; 

                        if (t_input >= 0.0 && t_input < t_final) {
                            // Capturar datos para inferencia
                            datos_inf.posicion = punto_interseccion; 
                            datos_inf.direccion = dir_mapped;
                            datos_inf.normal = norm_mapped; 
                            datos_inf.difuso = primitiva_intersectada->difuso; 
                            datos_inf.especular = primitiva_intersectada->especular;
                            
                            // Pasamos el tiempo real de iluminación a la red
                            datos_inf.tiempo = (float)t_input; 
                            // Guardamos el tiempo de la cámara en delta_t para reconstruir el tiempo total
                            datos_inf.delta_t = (float)tiempo_acumulado; 

                            throughput_inf = camino * reflectancia; 
                            
                            float max_th = 5.0f; 
                            throughput_inf.r = fminf(throughput_inf.r, max_th);
                            throughput_inf.g = fminf(throughput_inf.g, max_th);
                            throughput_inf.b = fminf(throughput_inf.b, max_th);

                            necesita_inf = true;
                        } else {
                            // En este instante T_target sorteado, la luz aún no ha llegado físicamente aquí.
                            necesita_inf = false;
                        }
                        if(modo_reconstruccion){
                            return color;
                        }
                        break;
                    } else if (es_ruta_entrenamiento && !punto_entrenamiento_encontrado) { // Capturar datos para entrenamiento
                        tiempo_camara_head = tiempo_acumulado;

                        registro_train.head.posicion = punto_interseccion; 
                        registro_train.head.direccion = dir_mapped;
                        registro_train.head.normal = norm_mapped; 
                        registro_train.head.difuso = primitiva_intersectada->difuso; 
                        registro_train.head.especular = primitiva_intersectada->especular;
                        throughput_at_train_vertex = camino * reflectancia;
                        color_acumulado_pre_train = color; 
                        a_prefix = a_current; // Guardar footprint del prefijo para comparar con el sufijo
                        punto_entrenamiento_encontrado = true;
                    }
                }

                // Luz indirecta
                calcularLuzIndirecta(rng_seed, rayo_actual, camino, r_inicial, 
                                punto_interseccion, normal, primitiva_intersectada, current_ior, pdf_ultimo_rebote, ultimo_fue_especular);

                // Ruleta rusa
                if(profundidad > 10) {
                    double prob = fmax(camino.r, fmax(camino.g, camino.b));
                    if (prob > 0.95) prob = 0.95;
                    if (pcg32_float(rng_seed) > prob) {
                        // Si estábamos en medio del sufijo, registramos el fin del rayo
                        if (es_ruta_entrenamiento && punto_entrenamiento_encontrado) {
                            // Tiempo desde la luz hasta el tail
                            double tiempo_luz_tail = fmax(0.0, tiempo_acumulado_NEE - tiempo_acumulado);
                            registro_train.tail.tiempo = (float)tiempo_luz_tail;
                            // Tiempo desde el head hasta el tail
                            double delta_t_head_tail = tiempo_acumulado - tiempo_camara_head;
                            // Tiempo desde el head hasta la luz (target para el MLP)
                            registro_train.head.tiempo = (float)(tiempo_luz_tail + delta_t_head_tail);

                            registro_train.luz_acumulada_sufijo = color - color_acumulado_pre_train;
                            registro_train.throughput_sufijo = Color(0,0,0); // El tail no aporta red porque acabó el rayo
                            registro_train.factor_normalizacion = throughput_at_train_vertex;
                            registro_train.valido = true;
                            guardar_train = true;
                        }
                        break;
                    }
                    camino = camino / prob;
                }
                        
            
            } else { // No hay intersección
                Color color_fondo(0, 0, 0);
                color = color + camino * color_fondo;
                if (es_ruta_entrenamiento && punto_entrenamiento_encontrado) {
                    // Si el sufijo sale de la escena, el tail es el fondo (negro)
                    registro_train.luz_acumulada_sufijo = color - color_acumulado_pre_train;
                    registro_train.throughput_sufijo = Color(0,0,0); 
                    registro_train.factor_normalizacion = throughput_at_train_vertex;
                    registro_train.valido = true;
                    guardar_train = true;
                }
                break;
            }
            profundidad++;
        }

        // Si el rayo alcanzó el max_depth y el sufijo nunca terminó, hay que cerrarlo
        if (es_ruta_entrenamiento && punto_entrenamiento_encontrado && !guardar_train) {
            // Tiempo desde la luz hasta el tail
            double tiempo_luz_tail = fmax(0.0, tiempo_acumulado_NEE - tiempo_acumulado);
            registro_train.tail.tiempo = (float)tiempo_luz_tail;
            // Tiempo desde el head hasta el tail
            double delta_t_head_tail = tiempo_acumulado - tiempo_camara_head;
            // Tiempo desde el head hasta la luz (target para el MLP)
            registro_train.head.tiempo = (float)(tiempo_luz_tail + delta_t_head_tail);

            registro_train.luz_acumulada_sufijo = color - color_acumulado_pre_train;
            registro_train.throughput_sufijo = Color(0,0,0); 
            registro_train.factor_normalizacion = throughput_at_train_vertex;
            registro_train.valido = true;
            guardar_train = true;
        }
        if (es_ruta_entrenamiento && punto_entrenamiento_encontrado) {
            color = color_acumulado_pre_train; // Descartamos la luz del sufijo para el píxel final
            necesita_inf = true;
            datos_inf.posicion = registro_train.head.posicion;
            datos_inf.direccion = registro_train.head.direccion;
            datos_inf.normal = registro_train.head.normal;
            datos_inf.difuso = registro_train.head.difuso;
            datos_inf.especular = registro_train.head.especular;
            datos_inf.tiempo = registro_train.head.tiempo;
            datos_inf.delta_t = (float)tiempo_camara_head;
            throughput_inf = throughput_at_train_vertex;
        }
        return color;
    }

    __device__ void calcularLuzIndirecta(uint32_t& rng_seed, Rayo& rayo_actual, Color& camino, const Rayo& r_inicial, 
                                 const Vector3d& punto_interseccion, const Vector3d& normal,
                                 const Primitiva* primitiva_intersectada, double& current_ior, float& pdf_salida, bool& ultimo_fue_especular) {

        Color k_d = primitiva_intersectada->difuso;
        Color k_s = primitiva_intersectada->especular;
        Color k_t = primitiva_intersectada->transmision;
        double max_kd = k_d.max();
        double max_ks = k_s.max();
        double max_kt = k_t.max();
        double suma_max = max_kd + max_ks + max_kt;

        // Evitar división por cero
        if (suma_max <= 0.0) {
            camino = Color(0,0,0);
            return;
        }

        double p_difuso = max_kd / suma_max;
        double p_especular = max_ks / suma_max;
        double p_transmision = 1 - p_difuso - p_especular;

        double rand_brdf = pcg32_float(rng_seed);
        
        if (rand_brdf < p_difuso) {
            // Actualizar camino
            camino = camino * k_d / p_difuso;

            // Muestrear dirección en el hemisferio usando distribución cosenoidal
            Vector3d wi_local = muestrearCosenoUniforme(rng_seed);

            // Asegurar que la dirección muestreada está en el hemisferio correcto
            while(wi_local.z <= EPSILON) {
                return;
            }

            Vector3d wi = transformarALMundo(wi_local, normal);

            // Lanzar rayo difuso (desplazar origen en la dirección del rayo)
            rayo_actual = Rayo(punto_interseccion + wi * EPSILON, wi);
            
            double cos_theta = abs(wi.dot(normal));
            // Protección para evitar división por cero en el footprint del siguiente rebote
            if (cos_theta < 1e-4) cos_theta = 1e-4; 
            pdf_salida = float(cos_theta / pi);

            ultimo_fue_especular = false;
            
        } else if (rand_brdf < p_difuso + p_especular) {
            // Componente especular
            Vector3d wi_especular = reflejar(rayo_actual.direccion(), normal);
            // Actualizar camino
            camino = camino * k_s / p_especular;
            // Preparar el siguiente rayo
            rayo_actual = Rayo(punto_interseccion + wi_especular * EPSILON, wi_especular);

            pdf_salida = 1e6f;

            ultimo_fue_especular = true;
        } else {
            // Componente de transmisión
            Vector3d wi_transmitido = refractar(rayo_actual.direccion(), normal, primitiva_intersectada->indice_refraccion, current_ior);
            // Actualizar camino
            camino = camino * k_t / p_transmision;
            // Preparar el siguiente rayo
            rayo_actual = Rayo(punto_interseccion + wi_transmitido * EPSILON, wi_transmitido);

            pdf_salida = 1e6f;

            ultimo_fue_especular = true;
        }
    }

    __device__ Vector3d reflejar(const Vector3d& rayo_direccion, const Vector3d& normal) {
        Vector3d r = rayo_direccion - 2.0 * normal * rayo_direccion.dot(normal);
        return r.normalized();
    }

    __device__ Vector3d refractar(const Vector3d& rayo_direccion, const Vector3d& normal, double indice_objeto, double& current_ior) {
        Vector3d I = rayo_direccion.normalized();
        Vector3d N = normal.normalized();

        double cosi = I.dot(N);                // Puede ser >0 si estamos dentro mirando hacia fuera
        cosi = fmax(-1.0, fmin(1.0, cosi));     // Clamp para estabilidad numérica

        double etai = current_ior;
        double etat;

        // Si cosi > 0 el rayo está saliendo del material
        if (cosi > 0.0) {
            N = -N;
            etat = 1.0; // Asumimos aire fuera
        } else {
            cosi = -cosi; // Hacemos cosi positivo para la fórmula
            etat = indice_objeto;
        }

        double eta = etai / etat;              // Relación de índices (n1 / n2)
        double k = 1.0 - eta * eta * (1.0 - cosi * cosi); // Discriminante

        if (k < 0.0) {
            // Reflexión interna total: devolver dirección reflejada
            // No cambiamos current_ior porque seguimos en el mismo medio
            return reflejar(I, N);
        }

        // Actualizamos el índice de refracción actual
        current_ior = etat;

        double cost = sqrt(k);                 // cos(theta_t)
        Vector3d T = eta * I + (eta * cosi - cost) * N;
        return T.normalized();
    }

    // Generar direcciones aleatorias en el hemisferio con distribución cosenoidal
    __device__ Vector3d muestrearCosenoUniforme(uint32_t& rng_seed) {
        double r1 = pcg32_float(rng_seed);  // para angulo azimutal
        double r2 = pcg32_float(rng_seed);  // para angulo polar

        // Coordenadas esféricas con distribución cosenoidal
        double phi = 2 * M_PI * r1; // angulo azimutal
        double theta = acos(sqrt(1 - r2)); // angulo polar

        // Convertir a coordenadas cartesianas (espacio local)
        double x = sin(theta) * cos(phi);
        double y = sin(theta) * sin(phi);
        double z = cos(theta);
        
        return Vector3d(x, y, z);
    }

    // Crear sistema de coordenadas ortonormal alrededor de la normal
    __device__ Vector3d transformarALMundo(const Vector3d& local, const Vector3d& normal) {
        Vector3d tangente;

        // Encontrar un vector perpendicular a la normal
        if (abs(normal.x) > abs(normal.y)) {
            tangente = Vector3d(normal.z, 0, -normal.x);
        } else {
            tangente = Vector3d(0, -normal.z, normal.y);
        }
        tangente.normalize();
        
        // Calcular el segundo vector tangente
        Vector3d bitangente = normal.cross(tangente);
        bitangente.normalize();
        
        // Transformar del espacio local al mundial
        return local.x * tangente + local.y * bitangente + local.z * normal;
    }

    // Método de renderizado para GPU
    __device__ void renderizar(const Camara& camara, const Escenario& escena, 
                              int ancho_img, int alto_img, int x, int y, ImagenGPU& imagen, TransientRender& transientRenderer,
                              int frame_number, 
                              // Datos para entrenamiento
                              unsigned int* dev_counter, 
                              RegistroEntrenamiento* buffer_training, unsigned int* counter_training, int max_cap_training,
                              // Datos para inferencia
                              DatosMLP* buffer_inference, Color* buffer_throughput, bool usar_red_inferencia, bool entrenar_red,
                              bool modo_reconstruccion) {
        
        // Inicializar semilla RNG basada en píxel, frame y profundidad
        uint32_t rng_seed = inicializarSemilla(x, y, frame_number);

        Color color_pixel(0, 0, 0);
        
        float offset_x = pcg32_float(rng_seed);
        float offset_y = pcg32_float(rng_seed);

        Rayo rayo = camara.generarRayo(x, y, ancho_img, alto_img, offset_x, offset_y);

        // Variables para entrenamiento e inferencia
        RegistroEntrenamiento reg_train;
        bool guardar_train = false;
        DatosMLP datos_inferencia;
        Color throughput_inferencia(0,0,0);
        bool necesita_inferencia = false;
        // Fase 1 (Warmup): entrenar_red=true, usar_red_inferencia=false → 100% entrenamiento
        // Fase 2 (Post-warmup a frame 80): entrenar_red=true, usar_red_inferencia=true → 3% entrena, 97% infiere
        // Fase 3 (Frame 80+): entrenar_red=false, usar_red_inferencia=true → 0% entrena, 100% infiere
        bool es_training_pixel = !modo_reconstruccion ? (entrenar_red ? (usar_red_inferencia ? (pcg32_float(rng_seed) < 0.2f) : true) : false) : false;

        Color color = lanzarRayoIterativo(rayo, escena, rng_seed, x, y, frame_number, transientRenderer,
                                            reg_train, guardar_train, 
                                            datos_inferencia, throughput_inferencia, necesita_inferencia, es_training_pixel, usar_red_inferencia, 
                                            modo_reconstruccion);

        imagen.setPixel(x, y, color);
        
        // Guardar Datos Inferencia
        if (necesita_inferencia) {
            buffer_inference[y * ancho_img + x] = datos_inferencia;
            buffer_throughput[y * ancho_img + x] = throughput_inferencia;
        }
        else {
            buffer_throughput[y * ancho_img + x] = Color(0,0,0);
        }

        // Guardar Datos Entrenamiento solo si es pixel de entrenamiento y se ha marcado para guardar
        if (es_training_pixel && guardar_train && counter_training != nullptr) {
            unsigned int idx = atomicAdd(counter_training, 1);
            buffer_training[idx % max_cap_training] = reg_train;
        }
    }
};

#endif
