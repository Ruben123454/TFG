#include "GuiLayer.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <KHR/glad.h>
#include <GLFW/glfw3.h>
#include <cstring>

bool GuiLayer::init(GLFWwindow* window, const char* glslVersion) {
	if (initialized_ || window == nullptr) {
		return false;
	}

	IMGUI_CHECKVERSION();
	context_ = ImGui::CreateContext();
	ImGui::SetCurrentContext(context_);

	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

	if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
		shutdown();
		return false;
	}

	if (!ImGui_ImplOpenGL3_Init(glslVersion)) {
		shutdown();
		return false;
	}

	initialized_ = true;
	return true;
}

void GuiLayer::shutdown() {
	if (!initialized_) {
		return;
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();

	if (context_ != nullptr) {
		ImGui::SetCurrentContext(context_);
		ImGui::DestroyContext(context_);
		context_ = nullptr;
	}

	initialized_ = false;
}

void GuiLayer::beginFrame() {
	if (!initialized_) {
		return;
	}

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void GuiLayer::drawConfigScreen(RenderGuiState& state) {
	if (!initialized_) {
		return;
	}

	ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
	ImGui::Begin("Configuracion Inicial", nullptr);

	ImGui::Text("=== MODO DE RENDERIZADO ===");
	ImGui::RadioButton("Modo Normal (Entrenamiento + Inferencia)", &state.renderMode, 0);
	ImGui::RadioButton("Modo Reconstruccion (Solo Inferencia)", &state.renderMode, 1);

	if (state.renderMode == 1) {
		ImGui::InputText("Ruta modelo MLP", state.mlpModelPath, IM_ARRAYSIZE(state.mlpModelPath));
	}

	ImGui::Separator();

	ImGui::Text("=== PARAMETROS DE IMAGEN ===");
	ImGui::InputInt("Ancho (px)##width", &state.imageWidth, 1, 100);
	if (state.imageWidth < 64) state.imageWidth = 64;
	if (state.imageWidth > 2048) state.imageWidth = 2048;
	
	ImGui::InputInt("Alto (px)##height", &state.imageHeight, 1, 100);
	if (state.imageHeight < 64) state.imageHeight = 64;
	if (state.imageHeight > 2048) state.imageHeight = 2048;
	
	ImGui::InputInt("Muestras por Pixel (SPP)##spp", &state.samplesPerPixel, 1, 10);
	if (state.samplesPerPixel < 1) state.samplesPerPixel = 1;
	if (state.samplesPerPixel > 1000000) state.samplesPerPixel = 1000000;

	ImGui::Checkbox("Activar Transient Render", &state.activarTransient);
	if (state.activarTransient) {
		ImGui::InputInt("Frames Transient", &state.transientFrames, 1, 10);
		if (state.transientFrames < 1) state.transientFrames = 1;
		if (state.transientFrames > 10000) state.transientFrames = 10000;
	}

	ImGui::Separator();
	ImGui::Text("=== MODELO 3D ===");
	ImGui::Checkbox("Cargar modelo externo", &state.loadModel);
	
	if (state.loadModel) {
		ImGui::InputText("Ruta del modelo", state.modelPath, IM_ARRAYSIZE(state.modelPath));
	}

	ImGui::Separator();
	ImGui::Text("=== SALIDA ===");
	ImGui::InputText("Nombre archivo salida", state.outputFileName, IM_ARRAYSIZE(state.outputFileName));
	if(!state.renderMode == 1) {
		ImGui::Checkbox("Guardar modelo MLP entrenado", &state.saveMlpModel);
		if (state.saveMlpModel) {
			ImGui::InputText("Ruta modelo MLP", state.mlpModelPath, IM_ARRAYSIZE(state.mlpModelPath));
		}
	}
	ImGui::Separator();
	ImGui::SetCursorPosX((ImGui::GetWindowWidth() - 120) * 0.5f);
	if (ImGui::Button("INICIAR RENDERIZADO", ImVec2(160, 40))) {
		state.readyToRender = true;
	}

	ImGui::End();
}

void GuiLayer::drawControls(RenderGuiState& state) {
	if (!initialized_) {
		return;
	}

	ImGui::Begin("Path Tracer Controls");

	ImGui::Text("SPP acumuladas: %d", state.accumulatedSamples);
	ImGui::Text("Tiempo frame: %.2f ms", state.lastFrameMs);
	ImGui::Text("Tiempo total render: %.2f ms", state.totalRenderMs);
	ImGui::Text("Tiempo total ejecución: %.2f ms", state.totalExecutionMs);
	if (state.warmupActive) {
		ImGui::Separator();
		ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "WARM UP");
		ImGui::TextUnformatted("Entrenando red... la imagen final aun no se acumula.");
	}
	ImGui::Separator();

	if (ImGui::Button(state.pauseRendering ? "Reanudar" : "Pausar")) {
		state.pauseRendering = !state.pauseRendering;
	}

	ImGui::SameLine();
	if (ImGui::Button("Reset acumulacion")) {
		state.resetAccumulation = true;
	}

	if(ImGui::InputInt("Ancho (px)##width", &state.imageWidth, 1, 100)) {
		state.configUpdate = true;
	}
	if (state.imageWidth < 64) state.imageWidth = 64;
	if (state.imageWidth > 2048) state.imageWidth = 2048;
	
	if(ImGui::InputInt("Alto (px)##height", &state.imageHeight, 1, 100)) {
		state.configUpdate = true;
	}
	if (state.imageHeight < 64) state.imageHeight = 64;
	if (state.imageHeight > 2048) state.imageHeight = 2048;
	
	if(ImGui::InputInt("SPP##spp", &state.samplesPerPixel, 1, 10)) {
		state.configUpdate = true;
	}
	if (state.samplesPerPixel < 1) state.samplesPerPixel = 1;
	if (state.samplesPerPixel > 1000000) state.samplesPerPixel = 1000000;

	if(!state.renderMode == 1) {
		ImGui::Separator();
		ImGui::Checkbox("Guardar modelo MLP entrenado", &state.saveMlpModel);
		if (state.saveMlpModel) {
			ImGui::InputText("Ruta modelo MLP", state.mlpModelPath, IM_ARRAYSIZE(state.mlpModelPath));
		}
	}

	if (state.renderingComplete) {
		ImGui::Separator();
		ImGui::TextUnformatted("Renderizado completado");
		if(!state.renderMode == 1) {
			if (state.saveMlpModel) {
				ImGui::Text("Modelo MLP guardado en: %s", state.mlpModelPath);
			} else {
				ImGui::TextUnformatted("Modelo MLP no guardado");
			}
		}
		ImGui::Text("Archivo: %s", state.outputFileName);
		ImGui::InputText("Nombre archivo salida", state.outputFileName, IM_ARRAYSIZE(state.outputFileName));
		if (ImGui::Button("Guardar imagen")) {
			state.requestSave = true;
			state.renderingComplete = false;
		}
		ImGui::SameLine();
		if (ImGui::Button("No guardar")) {
			state.requestSave = false;
			state.renderingComplete = false;
		}
	}
	ImGui::End();

	ImGui::Begin("Debug");
	if(state.bvh) {
		ImGui::Text("Iformación BVH:");
		ImGui::Text("BVH Nodes: %d", state.bvhNodes);
		ImGui::Text("BVH Primitives: %d", state.bvhPrimitives);
	} else {
		ImGui::Text("BVH no construido");
	}
	ImGui::Text("Pérdida entrenamiento: %.6f", state.trainingLoss);
	ImGui::End();
}

void GuiLayer::drawSaveScreen(RenderGuiState& state) {
	if (!initialized_) {
		return;
	}

	ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 150, ImGui::GetIO().DisplaySize.y * 0.5f - 50), ImGuiCond_Appearing);
	ImGui::SetNextWindowSize(ImVec2(300, 100), ImGuiCond_Appearing);
	
	ImGui::Begin("Renderizado Completado", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);

	ImGui::Text("El renderizado ha finalizado!");

	ImGui::Text("Archivo: %s", state.outputFileName);

	ImGui::Separator();

	float button_width = (ImGui::GetWindowWidth() - 20) / 2;
	
	if (ImGui::Button("Guardar imagen", ImVec2(button_width, 30))) {
		state.requestSave = true;
		state.renderingComplete = false;
	}

	ImGui::SameLine();

	if (ImGui::Button("Descartar", ImVec2(button_width, 30))) {
		state.renderingComplete = false;
	}

	ImGui::End();
}

void GuiLayer::endFrame(int framebufferWidth, int framebufferHeight) {
	if (!initialized_) {
		return;
	}

	ImGui::Render();
	glViewport(0, 0, framebufferWidth, framebufferHeight);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

