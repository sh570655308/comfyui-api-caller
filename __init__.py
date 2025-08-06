from .comfyui_api_caller import (
    ComfyUIAPIConfig,
    ComfyUIAPIInput,
    ComfyUIAPIOutputImage,
    ComfyUIAPIOutputLatent,
    ComfyUIAPIOutputText
)

NODE_CLASS_MAPPINGS = {
    "ComfyUIAPIConfig": ComfyUIAPIConfig,
    "ComfyUIAPIInput": ComfyUIAPIInput,
    "ComfyUIAPIOutputImage": ComfyUIAPIOutputImage,
    "ComfyUIAPIOutputLatent": ComfyUIAPIOutputLatent,
    "ComfyUIAPIOutputText": ComfyUIAPIOutputText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUIAPIConfig": "API Config",
    "ComfyUIAPIInput": "API Input",
    "ComfyUIAPIOutputImage": "API Output (Image)",
    "ComfyUIAPIOutputLatent": "API Output (Latent)",
    "ComfyUIAPIOutputText": "API Output (Text)"
}