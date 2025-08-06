# ComfyUI API Caller

[English](#english) | [中文](#中文)

---

## 中文

### 插件目的

随着AI模型的不断增大和工作流的日益复杂化，ComfyUI在运行时会花费大量时间在模型的加载和卸载上。这个插件特别适合拥有多台电脑的用户，通过将工作流分发到不同的机器上执行，可以：

- **减少模型加载时间**：在专用机器上预加载模型，避免重复加载
- **提高工作效率**：多台机器并行处理，显著缩短工作流执行时间
- **优化资源利用**：将不同类型的模型分配到不同的机器上，充分利用硬件资源
- **简化工作流程**：通过API调用实现跨机器的无缝工作流执行

### 功能特性

- **模块化设计**：独立的配置、输入和不同输出类型的节点
- **多类型输入支持**：自动检测和处理图像、潜空间和文本
- **潜空间文件传输**：通过ComfyUI的SaveLatent/LoadLatent机制完全支持潜空间传输
- **跨实例工作流**：在远程ComfyUI实例上无缝执行工作流

### 节点类型

#### 1. API Config
- 设置API URL和工作流路径
- 作为其他节点的配置中心

#### 2. API Input  
- 处理最多5个自适应输入，支持自动类型检测
- 将输入映射到目标工作流中的特定节点ID
- 支持图像、潜空间和文本数据

#### 3. API Output Nodes
- **API Output (Image)**：下载并返回图像输出
- **API Output (Latent)**：下载并处理潜空间文件（.latent格式）
- **API Output (Text)**：检索文本输出

### 工作流程

```
Input Node → API Config → Output Node(s)
     ↓           ↓             ↓
   [Data]   [Settings]   [Results]
```

### 安装

1. 将此仓库克隆或下载到您的ComfyUI `custom_nodes`目录：
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/sh570655308/comfyui-api-caller.git
   ```

2. 重启ComfyUI

3. 节点将出现在"API"类别中

### 使用方法

#### 基本设置

1. **添加API Config节点**：
   - 设置`api_url`（例如：`http://192.168.1.10:8188`）
   - 设置`workflow_path`（.json工作流文件的路径）

2. **添加API Input节点**（可选）：
   - 连接到API Config
   - 添加您的输入数据（图像、文本等）
   - 设置相应的`input_X_node_id`值

3. **添加输出节点**：
   - 连接到API Config
   - 设置`output_node_id`以匹配您工作流的输出节点

#### 潜空间处理

对于潜空间输入/输出：
- **输入**：插件自动将潜空间保存为.latent文件并上传
- **输出**：目标工作流应包含SaveLatent节点；插件下载并加载结果

这保留了完整的通道信息（16通道Flux、4通道SDXL等）。

### 示例

#### 图像处理
```
LoadImage → API Input → API Config → API Output (Image) → PreviewImage
```

#### 潜空间工作流
```  
KSampler → API Input → API Config → API Output (Latent) → VAEDecode
```

#### 多输出
```
API Config → API Output (Image)
          → API Output (Latent) 
          → API Output (Text)
```

### 系统要求

- ComfyUI
- Python包：`requests`、`torch`、`PIL`、`safetensors`

### 故障排除

#### 潜空间通道不匹配
确保您的目标工作流对潜空间输出使用SaveLatent节点。插件处理基于文件的潜空间传输以保留通道信息。

#### 连接问题
- 验证目标ComfyUI API是否可访问
- 检查防火墙设置
- 确保API URL格式正确（包含http://）

#### 工作流未找到
- 对工作流文件使用绝对路径
- 确保.json工作流文件存在且有效

### 贡献

欢迎贡献！请随时提交问题和拉取请求。

### 许可证

本项目为开源项目。请查看许可证文件了解详情。

---

## English

A ComfyUI custom node plugin that enables calling external ComfyUI instances via API to execute workflows remotely.

### Features

- **Modular Design**: Separate nodes for configuration, input, and different output types
- **Multi-type Input Support**: Automatic detection and handling of images, latents, and text
- **Latent File Transmission**: Full support for latent transfer via ComfyUI's SaveLatent/LoadLatent mechanism
- **Cross-Instance Workflows**: Execute workflows on remote ComfyUI instances seamlessly

### Node Types

#### 1. API Config
- Sets API URL and workflow path
- Acts as the configuration hub for other nodes

#### 2. API Input  
- Handles up to 5 adaptive inputs with automatic type detection
- Maps inputs to specific node IDs in the target workflow
- Supports images, latents, and text data

#### 3. API Output Nodes
- **API Output (Image)**: Downloads and returns image outputs
- **API Output (Latent)**: Downloads and processes latent files (.latent format)
- **API Output (Text)**: Retrieves text outputs

### Workflow

```
Input Node → API Config → Output Node(s)
     ↓           ↓             ↓
   [Data]   [Settings]   [Results]
```

### Installation

1. Clone or download this repository to your ComfyUI `custom_nodes` directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone https://github.com/sh570655308/comfyui-api-caller.git
   ```

2. Restart ComfyUI

3. The nodes will appear in the "API" category

### Usage

#### Basic Setup

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

#### Latent Handling

For latent inputs/outputs:
- **Input**: Plugin automatically saves latents to .latent files and uploads them
- **Output**: Target workflow should include SaveLatent nodes; plugin downloads and loads the results

This preserves full channel information (16-channel Flux, 4-channel SDXL, etc.)

### Examples

#### Image Processing
```
LoadImage → API Input → API Config → API Output (Image) → PreviewImage
```

#### Latent Workflow
```  
KSampler → API Input → API Config → API Output (Latent) → VAEDecode
```

#### Multi-Output
```
API Config → API Output (Image)
          → API Output (Latent) 
          → API Output (Text)
```

### Requirements

- ComfyUI
- Python packages: `requests`, `torch`, `PIL`, `safetensors`

### Troubleshooting

#### Latent Channel Mismatch
Make sure your target workflow uses SaveLatent nodes for latent outputs. The plugin handles file-based latent transmission to preserve channel information.

#### Connection Issues
- Verify the target ComfyUI API is accessible
- Check firewall settings
- Ensure the API URL format is correct (include http://)

#### Workflow Not Found
- Use absolute paths for workflow files
- Ensure the .json workflow file exists and is valid

### Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### License

This project is open source. Please check the license file for details.
