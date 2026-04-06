#ifndef VISUALIZADOR_H
#define VISUALIZADOR_H

#include <KHR/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "color.h"
#include "GuiLayer.h"

class Visualizador {
public:
    GLFWwindow* window = nullptr;
    GuiLayer gui;

private:
    GLuint texture = 0;
    std::vector<uint32_t> pixel_buffer;
    RenderGuiState gui_state;
    int width, height;
    bool running = true;

public:
    Visualizador(int w, int h, const char* titulo) : width(w), height(h) {
        if (!glfwInit()) {
            throw std::runtime_error("No se pudo inicializar GLFW");
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

        window = glfwCreateWindow(width, height, titulo, nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            throw std::runtime_error("No se pudo crear la ventana GLFW");
        }

        glfwMaximizeWindow(window);

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwDestroyWindow(window);
            glfwTerminate();
            throw std::runtime_error("No se pudo inicializar GLAD");
        }

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        pixel_buffer.resize(width * height);

        if (!gui.init(window, "#version 330")) {
            throw std::runtime_error("No se pudo inicializar ImGui");
        }
    }

    void configurarRenderTarget(int renderWidth, int renderHeight, const char* titulo = nullptr) {
        width = std::max(1, renderWidth);
        height = std::max(1, renderHeight);

        pixel_buffer.assign(width * height, 0u);

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        if (titulo != nullptr) {
            glfwSetWindowTitle(window, titulo);
        }
    }

    ~Visualizador() {
        gui.shutdown();

        if (texture != 0) {
            glDeleteTextures(1, &texture);
            texture = 0;
        }

        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }

        glfwTerminate();
    }

    bool procesarEventos() {
        glfwPollEvents();
        running = !glfwWindowShouldClose(window);
        return running;
    }

    RenderGuiState& uiState() {
        return gui_state;
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
    
    void actualizar(const std::vector<Color>& acumulador, int muestras_actuales) {
        float inv_muestras = 1.0f / muestras_actuales;

        #pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            Color c = acumulador[i] * inv_muestras;

            float r = acesFilmic(c.r);
            float g = acesFilmic(c.g);
            float b = acesFilmic(c.b);

            uint8_t ir = static_cast<uint8_t>(r * 255.0f);
            uint8_t ig = static_cast<uint8_t>(g * 255.0f);
            uint8_t ib = static_cast<uint8_t>(b * 255.0f);

            pixel_buffer[i] = (ir) | (ig << 8) | (ib << 16) | (255u << 24);
        }

        glBindTexture(GL_TEXTURE_2D, texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer.data());

        gui_state.accumulatedSamples = muestras_actuales;

        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(window, &fbw, &fbh);

        glViewport(0, 0, fbw, fbh);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        gui.beginFrame();
        gui.drawControls(gui_state);

        ImGui::Begin("Render");
        ImVec2 avail = ImGui::GetContentRegionAvail();
        float sx = avail.x / static_cast<float>(width);
        float sy = avail.y / static_cast<float>(height);
        float scale = std::max(0.1f, std::min(sx, sy));
        ImVec2 imageSize(width * scale, height * scale);
        ImGui::Image((ImTextureID)(intptr_t)texture, imageSize, ImVec2(0, 0), ImVec2(1, 1));

        if (gui_state.warmupActive) {
            ImVec2 imageMin = ImGui::GetItemRectMin();
            ImVec2 imageMax = ImGui::GetItemRectMax();
            ImVec2 center((imageMin.x + imageMax.x) * 0.5f, (imageMin.y + imageMax.y) * 0.5f);

            const char* warmupText = "WARM UP";
            ImVec2 textSize = ImGui::CalcTextSize(warmupText);

            ImDrawList* drawList = ImGui::GetWindowDrawList();
            drawList->AddText(
                ImVec2(center.x - textSize.x * 0.5f, center.y - textSize.y * 0.5f),
                IM_COL32(255, 191, 64, 255),
                warmupText
            );
        }
        ImGui::End();

        gui.endFrame(fbw, fbh);
        glfwSwapBuffers(window);
    }
};

#endif