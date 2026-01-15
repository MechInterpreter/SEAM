"""
Model loading for EdgePatch.

Loads the target model with 4-bit quantization and eager attention.
Pre-scans the model to tag attention modules with layer indices.
"""

import logging
from typing import Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from edgepatch.config import EdgePatchConfig

logger = logging.getLogger("edgepatch")


# Module names that are attention modules
ATTENTION_MODULE_NAMES = [
    "self_attn",  # Llama, DeepSeek
    "attention",  # Some models
]


def load_model_and_tokenizer(
    config: EdgePatchConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, dict]:
    """
    Load model and tokenizer with configuration from config.
    
    Returns:
        tuple of (model, tokenizer, model_info)
        model_info contains: num_layers, num_heads, layer_to_module mapping
    """
    logger.info(f"Loading model: {config.model_name}")
    
    # Quantization config for 4-bit
    if config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model with eager attention (REQUIRED for patching)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",  # CRITICAL: required for masking
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Pre-scan model to find attention modules and tag with layer indices
    model_info = tag_attention_layers(model)
    
    logger.info(
        f"Model loaded: {model_info['num_layers']} layers, "
        f"{model_info['num_heads']} heads per layer"
    )
    
    return model, tokenizer, model_info


def tag_attention_layers(model: AutoModelForCausalLM) -> dict:
    """
    Pre-scan model to find attention modules and tag them with layer indices.
    
    This creates a deterministic mapping from layer index to the actual
    attention module instance. Critical for scoped masking.
    
    Returns:
        dict with keys:
        - num_layers: total number of layers
        - num_heads: number of attention heads per layer
        - layer_to_module: dict mapping layer idx to attention module
        - module_to_layer: dict mapping module id to layer idx
    """
    layer_to_module = {}
    module_to_layer = {}
    num_heads = None
    
    # Find the transformer layers
    # Common patterns: model.model.layers, model.transformer.h, etc.
    layers = None
    
    # Try common paths
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # Llama, DeepSeek
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h  # GPT-2
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        layers = model.gpt_neox.layers  # GPT-NeoX
    
    if layers is None:
        raise RuntimeError("Could not find transformer layers in model")
    
    # Iterate through layers and find attention modules
    for layer_idx, layer in enumerate(layers):
        attn_module = None
        
        # Try common attention module names
        for name in ATTENTION_MODULE_NAMES:
            if hasattr(layer, name):
                attn_module = getattr(layer, name)
                break
        
        if attn_module is None:
            logger.warning(f"Could not find attention module in layer {layer_idx}")
            continue
        
        # Tag the module with its layer index
        attn_module._edgepatch_layer_idx = layer_idx
        
        layer_to_module[layer_idx] = attn_module
        module_to_layer[id(attn_module)] = layer_idx
        
        # Get number of heads from first attention module
        if num_heads is None:
            num_heads = _get_num_heads(attn_module)
    
    if not layer_to_module:
        raise RuntimeError("No attention modules found in model")
    
    return {
        "num_layers": len(layer_to_module),
        "num_heads": num_heads or 32,  # Fallback
        "layer_to_module": layer_to_module,
        "module_to_layer": module_to_layer,
    }


def _get_num_heads(attn_module) -> Optional[int]:
    """Extract number of attention heads from module."""
    # Try common attribute names
    for attr in ['num_heads', 'n_heads', 'num_attention_heads']:
        if hasattr(attn_module, attr):
            return getattr(attn_module, attr)
    
    # Try config
    if hasattr(attn_module, 'config'):
        config = attn_module.config
        for attr in ['num_attention_heads', 'n_head', 'num_heads']:
            if hasattr(config, attr):
                return getattr(config, attr)
    
    return None


def get_device(model: AutoModelForCausalLM) -> torch.device:
    """Get the device the model is on."""
    return next(model.parameters()).device
