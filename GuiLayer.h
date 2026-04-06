#pragma once

#include <cstdint>
#include "CudaInterop.h"

struct ImGuiContext;
struct GLFWwindow;

class GuiLayer {
public:
    GuiLayer() = default;
    ~GuiLayer() = default;

    bool init(GLFWwindow* window, const char* glslVersion = "#version 330");
    void shutdown();

    void beginFrame();
    void drawConfigScreen(RenderGuiState& state);
    void drawControls(RenderGuiState& state);
    void drawSaveScreen(RenderGuiState& state);
    void endFrame(int framebufferWidth, int framebufferHeight);

private:
    bool initialized_ = false;
    ImGuiContext* context_ = nullptr;
};
