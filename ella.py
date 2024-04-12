import os
import torch
import comfy

import folder_paths
from folder_paths import supported_pt_extensions, models_dir, folder_names_and_paths
from .ella_model.model import ELLA, T5TextEmbedder

folder_names_and_paths["ella"] = ([os.path.join(models_dir, "ella")], supported_pt_extensions)
folder_names_and_paths["t5_model"] = ([os.path.join(models_dir, "t5_model")],[])

for i, f_path in enumerate([folder_names_and_paths["t5_model"], folder_names_and_paths["ella"]]):
    f_path = f_path[0][0]

    if i == 0:
        f_path = f_path + "/flan-t5-xl"
        
    if not os.path.exists(f_path):
        os.makedirs(f_path, exist_ok=True)

# Borrowed from https://github.com/BlenderNeko/ComfyUI_Noise Until patch update.
class GetSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "sigmas": ("SIGMAS",),
            "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
            }}
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "calc_sigma"

    CATEGORY = "latent/noise"
        
    def calc_sigma(self, model, sigmas, start_at_step, end_at_step):
        end_at_step = min(len(sigmas) - 1, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        return (sigma.item(),)

class LoadElla:
    def __init__(self):
        self.device = comfy.model_management.text_encoder_device()
        self.dtype = comfy.model_management.text_encoder_dtype()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ella_model": (folder_paths.get_filename_list("ella"),),
                              "t5_model": (os.listdir(folder_names_and_paths["t5_model"][0][0]),),
                              }}

    RETURN_TYPES = ("ELLA",)
    FUNCTION = "load_ella"

    CATEGORY = "ella/loaders"

    def load_ella(self, ella_model, t5_model):
        t5_path = os.path.join(models_dir, 't5_model', t5_model)
        ella_path = os.path.join(models_dir, 'ella', ella_model)
        t5_model = T5TextEmbedder(t5_path).to(self.device, self.dtype)
        ella = ELLA().to(self.device, self.dtype)

        ella_state_dict = comfy.utils.load_torch_file(ella_path)

        ella.load_state_dict(ella_state_dict)

        return ({"ELLA": ella, "T5": t5_model}, )

class ELLATextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}), 
                "sigma": ("FLOAT", {"default": 1000.0}, ),
                "ella_dict": ("ELLA", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "ella/conditioning"

    def encode(self, text, sigma: float, ella_dict: dict):
        ella: ELLA = ella_dict.get("ELLA")
        t5: T5TextEmbedder = ella_dict.get("T5")

        cond = t5(text)
        cond_ella = ella(cond, timesteps=torch.tensor(sigma))
        
        return ([[cond_ella, {"pooled_output": cond_ella}]], ) # Output twice as we don't use pooled output
        
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadElla": LoadElla,
    "ELLATextEncode": ELLATextEncode,
    "GetSigma": GetSigma
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadElla": "Load ELLA Model",
    "ELLATextEncode": "ELLA Text Encode (Prompt)",
    "GetSigma": "Get Sigma (BNK)"
}
