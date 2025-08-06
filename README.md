# ComfyUI API Caller

A ComfyUI custom node plugin that enables calling external ComfyUI instances via API to execute workflows remotely.

## Features

- **Modular Design**: Separate nodes for configuration, input, and different output types
- **Multi-type Input Support**: Automatic detection and handling of images, latents, and text
- **Latent File Transmission**: Full support for latent transfer via ComfyUI's SaveLatent/LoadLatent mechanism
- **Cross-Instance Workflows**: Execute workflows on remote ComfyUI instances seamlessly

## Node Types

### 1. API Config
- Sets API URL and workflow path
- Acts as the configuration hub for other nodes

### 2. API Input  
- Handles up to 5 adaptive inputs with automatic type detection
- Maps inputs to specific node IDs in the target workflow
- Supports images, latents, and text data

### 3. API Output Nodes
- **API Output (Image)**: Downloads and returns image outputs
- **API Output (Latent)**: Downloads and processes latent files (.latent format)
- **API Output (Text)**: Retrieves text outputs

## Workflow

```
Input Node → API Config → Output Node(s)
     ↓           ↓             ↓
   [Data]   [Settings]   [Results]
```

## Installation

1. Clone or download this repository to your ComfyUI `custom_nodes` directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/sh570655308/comfyui-api-caller.git
   ```

2. Restart ComfyUI

3. The nodes will appear in the "API" category

## Usage

### Basic Setup

1. **Add API Config Node**: 
   - Set `api_url` (e.g., `http://192.168.1.10:8188`)
   - Set `workflow_path` (path to your .json workflow file)

2. **Add API Input Node** (optional):
   - Connect to API Config
   - Add your input data (images, text, etc.)
   - Set corresponding `input_X_node_id` values

3. **Add Output Nodes**:
   - Connect to API Config 
   - Set `output_node_id` to match your workflow's output nodes

### Latent Handling

For latent inputs/outputs:
- **Input**: Plugin automatically saves latents to .latent files and uploads them
- **Output**: Target workflow should include SaveLatent nodes; plugin downloads and loads the results

This preserves full channel information (16-channel Flux, 4-channel SDXL, etc.)

## Examples

### Image Processing
```
LoadImage → API Input → API Config → API Output (Image) → PreviewImage
```

### Latent Workflow
```  
KSampler → API Input → API Config → API Output (Latent) → VAEDecode
```

### Multi-Output
```
API Config → API Output (Image)
          → API Output (Latent) 
          → API Output (Text)
```

## Requirements

- ComfyUI
- Python packages: `requests`, `torch`, `PIL`, `safetensors`

## Troubleshooting

### Latent Channel Mismatch
Make sure your target workflow uses SaveLatent nodes for latent outputs. The plugin handles file-based latent transmission to preserve channel information.

### Connection Issues
- Verify the target ComfyUI API is accessible
- Check firewall settings
- Ensure the API URL format is correct (include http://)

### Workflow Not Found
- Use absolute paths for workflow files
- Ensure the .json workflow file exists and is valid

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source. Please check the license file for details.