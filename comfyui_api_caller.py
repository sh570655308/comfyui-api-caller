import json
import requests
import time
import io
import os
from PIL import Image
import torch
import numpy as np

try:
    from comfy.comfy_types.node_typing import IO
except ImportError:
    # Fallback if IO is not available
    class IO:
        ANY = "*"

# Custom type for API data flow
COMFYUI_API_TYPE = "COMFYUI_API"

class ComfyUIAPIConfig:
    """
    API Configuration node - sets API URL and workflow path
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:8188",
                    "multiline": False
                }),
                "workflow_path": ("STRING", {
                    "default": "C:/path/to/workflow.json",
                    "multiline": False
                }),
                "timeout_seconds": ("INT", {
                    "default": 300,
                    "min": 10,
                    "max": 3600,
                    "step": 10,
                    "display": "number"
                })
            }
        }
    
    RETURN_TYPES = (COMFYUI_API_TYPE,)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "create_config"
    CATEGORY = "API"
    
    def create_config(self, api_url, workflow_path, timeout_seconds):
        """Create API configuration"""
        config = {
            "api_url": api_url,
            "workflow_path": workflow_path,
            "timeout_seconds": timeout_seconds,
            "inputs": {},
            "executed": False,
            "outputs": None
        }
        return (config,)


class ComfyUIAPIInput:
    """
    API Input node - handles 5 adaptive inputs with node ID mapping
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": (COMFYUI_API_TYPE,)
            },
            "optional": {
                "input_1": (IO.ANY,),
                "input_1_node_id": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
                "input_2": (IO.ANY,),
                "input_2_node_id": ("INT", {"default": 2, "min": 1, "max": 9999, "step": 1}),
                "input_3": (IO.ANY,),
                "input_3_node_id": ("INT", {"default": 3, "min": 1, "max": 9999, "step": 1}),
                "input_4": (IO.ANY,),
                "input_4_node_id": ("INT", {"default": 4, "min": 1, "max": 9999, "step": 1}),
                "input_5": (IO.ANY,),
                "input_5_node_id": ("INT", {"default": 5, "min": 1, "max": 9999, "step": 1})
            }
        }
    
    RETURN_TYPES = (COMFYUI_API_TYPE,)
    RETURN_NAMES = ("api_config_with_inputs",)
    FUNCTION = "add_inputs"
    CATEGORY = "API"
    
    def detect_input_type(self, input_data):
        """Detect the type of input data"""
        if input_data is None:
            return None
        elif torch.is_tensor(input_data):
            if input_data.dim() >= 3:  # Image tensor
                return "image"
            else:  # Latent tensor
                return "latent"
        elif isinstance(input_data, dict) and "samples" in input_data:
            return "latent"
        elif isinstance(input_data, (int, float)):
            return "number"
        elif isinstance(input_data, str):
            return "text"
        else:
            # Try to convert to string as fallback
            return "text"
    
    def add_inputs(self, api_config, 
                   input_1=None, input_1_node_id=1,
                   input_2=None, input_2_node_id=2,
                   input_3=None, input_3_node_id=3,
                   input_4=None, input_4_node_id=4,
                   input_5=None, input_5_node_id=5):
        """Add inputs to API configuration"""
        
        # Copy the config to avoid modifying the original
        new_config = api_config.copy()
        new_config["inputs"] = {}
        
        # Process each input
        inputs = [
            (input_1, input_1_node_id),
            (input_2, input_2_node_id),
            (input_3, input_3_node_id),
            (input_4, input_4_node_id),
            (input_5, input_5_node_id)
        ]
        
        for input_data, node_id in inputs:
            if input_data is not None:
                input_type = self.detect_input_type(input_data)
                if input_type:
                    new_config["inputs"][node_id] = {
                        "type": input_type,
                        "data": input_data
                    }
                    print(f"Added input for node {node_id}: type={input_type}, data={input_data if input_type == 'number' else str(input_data)[:50]}")
                else:
                    print(f"Failed to detect type for input {node_id}: {type(input_data)}, value={input_data}")
        
        return (new_config,)


class ComfyUIAPIBase:
    """
    Base class for API execution functionality
    """
    def load_workflow(self, workflow_path):
        """Load workflow JSON file"""
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            return workflow
        except Exception as e:
            print(f"Error loading workflow: {e}")
            return None
    
    def tensor_to_pil(self, tensor):
        """Convert tensor image to PIL Image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).byte()
        np_image = tensor.cpu().numpy()
        return Image.fromarray(np_image)
    
    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor"""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_image)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
    
    def latent_to_base64(self, latent_data):
        """Convert latent data to base64 encoded string"""
        try:
            if isinstance(latent_data, dict) and "samples" in latent_data:
                samples = latent_data["samples"]
                if torch.is_tensor(samples):
                    # Convert to numpy and then to bytes
                    numpy_array = samples.cpu().numpy()
                    
                    # Create metadata
                    metadata = {
                        "shape": list(numpy_array.shape),
                        "dtype": str(numpy_array.dtype)
                    }
                    
                    # Serialize numpy array to bytes
                    import pickle
                    array_bytes = pickle.dumps(numpy_array)
                    
                    # Encode to base64
                    array_base64 = base64.b64encode(array_bytes).decode('utf-8')
                    
                    return {
                        "data": array_base64,
                        "metadata": metadata,
                        "type": "latent_base64"
                    }
            return None
        except Exception as e:
            print(f"Error converting latent to base64: {e}")
            return None
    
    def base64_to_latent(self, base64_data):
        """Convert base64 encoded string back to latent data"""
        try:
            if isinstance(base64_data, dict) and base64_data.get("type") == "latent_base64":
                # Decode from base64
                array_bytes = base64.b64decode(base64_data["data"])
                
                # Deserialize numpy array
                import pickle
                numpy_array = pickle.loads(array_bytes)
                
                # Convert back to tensor
                samples_tensor = torch.from_numpy(numpy_array)
                
                return {"samples": samples_tensor}
            return None
        except Exception as e:
            print(f"Error converting base64 to latent: {e}")
            return None
    
    def execute_workflow(self, api_config):
        """Execute the workflow via ComfyUI API"""
        if api_config["executed"]:
            return api_config["outputs"]
        
        # Load workflow
        workflow = self.load_workflow(api_config["workflow_path"])
        if workflow is None:
            return None
        
        # Update workflow with inputs
        print(f"Updating workflow with {len(api_config['inputs'])} inputs")
        for node_id, input_info in api_config["inputs"].items():
            input_type = input_info["type"]
            input_data = input_info["data"]
            node_id_str = str(node_id)
            
            print(f"Processing input for node {node_id}: type={input_type}")
            
            if node_id_str in workflow:
                if "inputs" in workflow[node_id_str]:
                    node_inputs = workflow[node_id_str]["inputs"]
                    print(f"Node {node_id} found in workflow. Available inputs: {list(node_inputs.keys())}")
                else:
                    print(f"Node {node_id} found but has no 'inputs' field")
                    continue
            else:
                print(f"Node {node_id} not found in workflow. Available nodes: {list(workflow.keys())[:10]}...")
                continue
            
            if input_type == "text":
                # Try common text input field names
                text_fields = ["text", "prompt", "string", "positive", "negative"]
                updated = False
                for field_name in text_fields:
                    if field_name in node_inputs:
                        node_inputs[field_name] = str(input_data)
                        updated = True
                        print(f"Updated text field '{field_name}' in node {node_id} with: {str(input_data)[:50]}...")
                        break
                
                if not updated:
                    print(f"Warning: No recognized text field found in node {node_id}. Available fields: {list(node_inputs.keys())}")
            
            elif input_type == "number":
                # Try common number input field names
                number_fields = ["seed", "steps", "cfg", "width", "height", "batch_size", "denoise", 
                               "strength", "value", "amount", "scale", "factor", "multiplier"]
                updated = False
                for field_name in number_fields:
                    if field_name in node_inputs:
                        if isinstance(input_data, int):
                            node_inputs[field_name] = int(input_data)
                        else:
                            node_inputs[field_name] = float(input_data)
                        updated = True
                        print(f"Updated number field '{field_name}' in node {node_id} with: {input_data}")
                        break
                
                if not updated:
                    print(f"Warning: No recognized number field found in node {node_id}. Available fields: {list(node_inputs.keys())}")
                    # As fallback, try to update the first field that looks like it accepts numbers
                    for field_name, field_value in node_inputs.items():
                        if isinstance(field_value, (int, float)):
                            if isinstance(input_data, int):
                                node_inputs[field_name] = int(input_data)
                            else:
                                node_inputs[field_name] = float(input_data)
                            print(f"Updated fallback number field '{field_name}' in node {node_id} with: {input_data}")
                            break
            
            elif input_type == "image":
                # Upload image first
                try:
                    pil_image = self.tensor_to_pil(input_data)
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    upload_url = f"{api_config['api_url']}/upload/image"
                    files = {'image': ('input.png', img_byte_arr, 'image/png')}
                    response = requests.post(upload_url, files=files)
                    
                    if response.status_code == 200:
                        upload_result = response.json()
                        filename = upload_result.get("name")
                        if "image" in node_inputs:
                            node_inputs["image"] = filename
                        print(f"Updated image field in node {node_id} with filename: {filename}")
                    else:
                        print(f"Failed to upload image for node {node_id}: HTTP {response.status_code}")
                except Exception as e:
                    print(f"Error uploading image for node {node_id}: {e}")
            
            elif input_type == "latent":
                # For latent inputs, we'll save the latent to file and modify the workflow
                # to use LoadLatent node instead
                print(f"Processing latent input for node {node_id}")
                try:
                    # Save latent to a temporary file
                    import tempfile
                    import uuid
                    latent_filename = f"api_transfer_{uuid.uuid4().hex}.latent"
                    
                    # Use ComfyUI's latent saving mechanism
                    latent_samples = input_data["samples"]
                    output = {
                        "latent_tensor": latent_samples.contiguous(),
                        "latent_format_version_0": torch.tensor([])
                    }
                    
                    # Save to temporary location first
                    temp_file = os.path.join(tempfile.gettempdir(), latent_filename)
                    import comfy.utils
                    comfy.utils.save_torch_file(output, temp_file, metadata=None)
                    
                    print(f"Saved latent to temporary file: {temp_file}")
                    print(f"Latent shape: {latent_samples.shape}")
                    
                    # Upload the latent file to target ComfyUI
                    upload_url = f"{api_config['api_url']}/upload/image"  # ComfyUI uses same endpoint for all files
                    
                    with open(temp_file, 'rb') as f:
                        files = {'image': (latent_filename, f, 'application/octet-stream')}
                        response = requests.post(upload_url, files=files)
                    
                    if response.status_code == 200:
                        upload_result = response.json()
                        uploaded_filename = upload_result.get("name", latent_filename)
                        print(f"Successfully uploaded latent file: {uploaded_filename}")
                        
                        # Modify workflow to inject a LoadLatent node
                        load_latent_node_id = f"{node_id}_latent_loader"
                        workflow[load_latent_node_id] = {
                            "inputs": {
                                "latent": uploaded_filename
                            },
                            "class_type": "LoadLatent",
                            "_meta": {
                                "title": f"Load Latent for Node {node_id}"
                            }
                        }
                        
                        # Update the original node to connect to the LoadLatent node
                        if "inputs" in workflow.get(node_id_str, {}):
                            for input_key in ["samples", "latent"]:
                                if input_key in node_inputs:
                                    # Connect to the LoadLatent node output
                                    node_inputs[input_key] = [load_latent_node_id, 0]
                                    break
                        
                        print(f"Modified workflow to load latent from {uploaded_filename}")
                    else:
                        print(f"Failed to upload latent file: {response.status_code}")
                        print(f"Response: {response.text}")
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"Error processing latent input for node {node_id}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Execute workflow
        try:
            prompt_url = f"{api_config['api_url']}/prompt"
            prompt_data = {
                "prompt": workflow,
                "client_id": "comfyui_api_caller"
            }
            
            print(f"Submitting workflow to {prompt_url}")
            response = requests.post(prompt_url, json=prompt_data, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"Failed to submit workflow to API. Status: {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg += f"\nAPI Error: {error_detail['error']}"
                    if "node_errors" in error_detail:
                        error_msg += f"\nNode Errors: {error_detail['node_errors']}"
                except:
                    error_msg += f"\nResponse: {response.text}"
                
                print(error_msg)
                raise RuntimeError(error_msg)
            
            result = response.json()
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                error_msg = "No prompt_id received from API"
                print(error_msg)
                raise RuntimeError(error_msg)
            
            print(f"Workflow submitted successfully. Prompt ID: {prompt_id}")
            print(f"Waiting for completion (timeout: {api_config['timeout_seconds']} seconds)...")
            
            # Wait for completion with user-configured timeout
            history_url = f"{api_config['api_url']}/history/{prompt_id}"
            timeout_seconds = api_config.get('timeout_seconds', 300)
            check_interval = min(2, timeout_seconds / 10)  # Check every 2 seconds or 1/10 of timeout
            max_attempts = int(timeout_seconds / check_interval)
            
            for attempt in range(max_attempts):
                time.sleep(check_interval)
                
                try:
                    response = requests.get(history_url, timeout=10)
                    
                    if response.status_code == 200:
                        history = response.json()
                        if prompt_id in history:
                            status = history[prompt_id].get("status", {})
                            
                            # Check for completion
                            if status.get("completed", False):
                                outputs = history[prompt_id].get("outputs", {})
                                print(f"Workflow completed successfully after {(attempt + 1) * check_interval:.1f} seconds")
                                # Cache the outputs
                                api_config["outputs"] = outputs
                                api_config["executed"] = True
                                return outputs
                            
                            # Check for errors
                            elif "error" in status:
                                error_detail = status["error"]
                                error_msg = f"Workflow execution failed on remote ComfyUI:\n{error_detail}"
                                
                                # Try to get more detailed error information
                                if "exception_type" in error_detail:
                                    error_msg += f"\nException Type: {error_detail['exception_type']}"
                                if "exception_message" in error_detail:
                                    error_msg += f"\nException Message: {error_detail['exception_message']}"
                                if "traceback" in error_detail:
                                    error_msg += f"\nTraceback: {error_detail['traceback']}"
                                if "node_id" in error_detail:
                                    error_msg += f"\nFailed Node ID: {error_detail['node_id']}"
                                if "node_type" in error_detail:
                                    error_msg += f"\nFailed Node Type: {error_detail['node_type']}"
                                
                                print(error_msg)
                                raise RuntimeError(error_msg)
                            
                            # Still executing, continue waiting
                            else:
                                elapsed_time = (attempt + 1) * check_interval
                                if attempt % 10 == 0:  # Print progress every ~20 seconds
                                    print(f"Still waiting... ({elapsed_time:.1f}/{timeout_seconds} seconds)")
                    else:
                        print(f"Warning: Failed to check workflow status (HTTP {response.status_code})")
                        
                except requests.RequestException as e:
                    print(f"Warning: Network error while checking status: {e}")
                    continue
            
            # Timeout reached
            error_msg = f"Workflow execution timed out after {timeout_seconds} seconds"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        except requests.RequestException as e:
            error_msg = f"Network error when calling ComfyUI API: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during API execution: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)


class ComfyUIAPIOutputImage(ComfyUIAPIBase):
    """
    API Image Output node - outputs single image
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": (COMFYUI_API_TYPE,),
                "output_node_id": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 9999,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_image_output"
    CATEGORY = "API"
    
    def get_image_from_api_output(self, api_url, output_data):
        """Get image from API output"""
        try:
            print(f"Debug: API output data for image: {output_data}")
            
            if isinstance(output_data, dict) and "images" in output_data:
                images_info = output_data["images"]
                if images_info:
                    image_info = images_info[0]
                    filename = image_info.get("filename")
                    subfolder = image_info.get("subfolder", "")
                    folder_type = image_info.get("type", "output")
                    
                    print(f"Debug: Image info - filename: {filename}, subfolder: {subfolder}, type: {folder_type}")
                    
                    if subfolder:
                        image_url = f"{api_url}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
                    else:
                        image_url = f"{api_url}/view?filename={filename}&type={folder_type}"
                    
                    print(f"Debug: Requesting image from: {image_url}")
                    
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        pil_image = Image.open(io.BytesIO(response.content))
                        print(f"Debug: Successfully loaded image with size: {pil_image.size}")
                        return self.pil_to_tensor(pil_image)
                    else:
                        print(f"Debug: Failed to download image, status: {response.status_code}")
            
            return None
        except Exception as e:
            print(f"Error getting image from API output: {e}")
            return None
    
    def get_image_output(self, api_config, output_node_id):
        """Get image output from API"""
        outputs = self.execute_workflow(api_config)
        
        if outputs is None:
            # Return empty image
            empty_image = torch.zeros((1, 64, 64, 3))
            return (empty_image,)
        
        output_node_str = str(output_node_id)
        if output_node_str in outputs:
            image = self.get_image_from_api_output(api_config["api_url"], outputs[output_node_str])
            if image is not None:
                return (image,)
        
        # Return empty image if not found
        empty_image = torch.zeros((1, 64, 64, 3))
        return (empty_image,)


class ComfyUIAPIOutputLatent(ComfyUIAPIBase):
    """
    API Latent Output node - outputs single latent
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": (COMFYUI_API_TYPE,),
                "output_node_id": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 9999,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "get_latent_output"
    CATEGORY = "API"
    
    def get_latent_from_api_output(self, api_url, output_data):
        """Get latent from API output by downloading latent files"""
        try:
            print(f"Debug: API output data for latent: {output_data}")
            
            # Check if output contains latent files (from SaveLatent node)
            if isinstance(output_data, dict) and "latents" in output_data:
                latents_info = output_data["latents"]
                if latents_info and len(latents_info) > 0:
                    # Get the first latent file
                    latent_info = latents_info[0]
                    filename = latent_info.get("filename")
                    subfolder = latent_info.get("subfolder", "")
                    folder_type = latent_info.get("type", "output")
                    
                    print(f"Debug: Latent file info - filename: {filename}, subfolder: {subfolder}, type: {folder_type}")
                    
                    # Download the latent file
                    if subfolder:
                        latent_url = f"{api_url}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
                    else:
                        latent_url = f"{api_url}/view?filename={filename}&type={folder_type}"
                    
                    print(f"Debug: Downloading latent from: {latent_url}")
                    
                    response = requests.get(latent_url)
                    if response.status_code == 200:
                        # Save to temporary file and load using ComfyUI's method
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.latent', delete=False) as tmp_file:
                            tmp_file.write(response.content)
                            tmp_file.flush()
                            
                            try:
                                # Load the latent file using ComfyUI's mechanism
                                import safetensors.torch
                                latent_data = safetensors.torch.load_file(tmp_file.name, device="cpu")
                                
                                # Apply the same logic as LoadLatent node
                                multiplier = 1.0
                                if "latent_format_version_0" not in latent_data:
                                    multiplier = 1.0 / 0.18215
                                
                                samples = {"samples": latent_data["latent_tensor"].float() * multiplier}
                                print(f"Debug: Successfully loaded latent with shape: {samples['samples'].shape}")
                                
                                return samples
                                
                            finally:
                                # Clean up temp file
                                try:
                                    os.remove(tmp_file.name)
                                except:
                                    pass
                    else:
                        print(f"Debug: Failed to download latent file, status: {response.status_code}")
            
            # If no latent files found, check for other output types
            if isinstance(output_data, dict):
                if "images" in output_data:
                    print("Found image output - you may want to use the Image output node instead")
                elif "text" in output_data:
                    print("Found text output - you may want to use the Text output node instead")
                else:
                    print("No recognized output format found. Make sure your workflow includes a SaveLatent node.")
            
            return None
        except Exception as e:
            print(f"Error processing latent API output: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_latent_output(self, api_config, output_node_id):
        """Get latent output from API"""
        outputs = self.execute_workflow(api_config)
        
        if outputs is None:
            # Return empty latent with minimal channels for compatibility
            empty_latent = {"samples": torch.zeros((1, 4, 64, 64))}
            return (empty_latent,)
        
        output_node_str = str(output_node_id)
        if output_node_str in outputs:
            latent = self.get_latent_from_api_output(api_config["api_url"], outputs[output_node_str])
            if latent is not None:
                print(f"Debug: Successfully retrieved latent with shape: {latent['samples'].shape}")
                return (latent,)
            else:
                print(f"Debug: Could not extract latent from output node {output_node_id}")
        else:
            print(f"Debug: Output node {output_node_id} not found in API outputs")
            print(f"Debug: Available output nodes: {list(outputs.keys())}")
        
        # Return empty latent if not found 
        empty_latent = {"samples": torch.zeros((1, 4, 64, 64))}
        return (empty_latent,)


class ComfyUIAPIOutputText(ComfyUIAPIBase):
    """
    API Text Output node - outputs single text
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": (COMFYUI_API_TYPE,),
                "output_node_id": ("INT", {
                    "default": 9,
                    "min": 1,
                    "max": 9999,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_text_output"
    CATEGORY = "API"
    
    def get_text_output(self, api_config, output_node_id):
        """Get text output from API"""
        outputs = self.execute_workflow(api_config)
        
        if outputs is None:
            return ("",)
        
        output_node_str = str(output_node_id)
        if output_node_str in outputs:
            # Try to extract text from output - this depends on the node type
            output_data = outputs[output_node_str]
            if isinstance(output_data, dict):
                # Look for common text fields
                if "text" in output_data:
                    return (str(output_data["text"]),)
                elif "string" in output_data:
                    return (str(output_data["string"]),)
            elif isinstance(output_data, (str, list)):
                return (str(output_data),)
        
        return ("",)