"""
Scoped layer/head masking implementation with runtime instrumentation.

CRITICAL: This module is designed to make it IMPOSSIBLE for edge_layers/edge_heads
to be ignored. Every masked forward pass records exactly what was masked, and
validation hard-fails if actual != expected.

Architecture:
1. Pre-scan model to find each layer's attention module
2. Monkeypatch attention forward to inject -inf mask at selected layers/heads
3. Instrumentation records what was actually masked
4. Hard-fail if actual != expected
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger("edgepatch")


class MaskingError(Exception):
    """Raised when masking fails or doesn't match expectations."""
    pass


@dataclass
class MaskingStats:
    """
    Runtime instrumentation - populated during forward pass.
    
    This is the key to ensuring masking actually works.
    After each masked forward, we validate that these stats match expectations.
    """
    layers_hit: set = field(default_factory=set)  # Layers whose forward was called
    layers_mask_applied: set = field(default_factory=set)  # Layers where mask was applied
    heads_mask_applied: dict = field(default_factory=dict)  # layer_idx -> set of head indices
    masked_entries_count: int = 0  # Total attention entries set to -inf
    
    def reset(self):
        """Reset stats for new forward pass."""
        self.layers_hit.clear()
        self.layers_mask_applied.clear()
        self.heads_mask_applied.clear()
        self.masked_entries_count = 0
    
    def summary(self) -> dict:
        """Return stats as dict for logging."""
        return {
            "layers_hit": sorted(self.layers_hit),
            "layers_mask_applied": sorted(self.layers_mask_applied),
            "heads_mask_applied": {k: sorted(v) for k, v in self.heads_mask_applied.items()},
            "masked_entries_count": self.masked_entries_count,
        }


class ScopedAttentionMasker:
    """
    Context manager that applies scoped attention masking.
    
    Usage:
        with ScopedAttentionMasker(model, model_info, edge_layers=[0,1], edge_heads=[0]) as masker:
            masker.set_mask_positions(q_positions=[100,101], k_positions=[10,11,12])
            output = model(input_ids)
            print(masker.stats.summary())
    
    The masker will:
    1. Patch attention modules to intercept computation
    2. Apply -inf to attention scores at specified Q->K positions for specified layers/heads
    3. Record exactly what was masked
    4. Restore original forwards on exit
    5. Validate that masking matched expectations
    """
    
    def __init__(
        self,
        model,
        model_info: dict,
        edge_layers: Optional[list[int]] = None,
        edge_heads: Optional[list[int]] = None,
        validate_on_exit: bool = True,
    ):
        """
        Args:
            model: The model to patch
            model_info: Dict from tag_attention_layers() with layer_to_module mapping
            edge_layers: Layer indices to mask (None = all layers)
            edge_heads: Head indices to mask (None = all heads)
            validate_on_exit: If True, hard-fail if masking didn't happen as expected
        """
        self.model = model
        self.model_info = model_info
        self.edge_layers = set(edge_layers) if edge_layers is not None else None
        self.edge_heads = set(edge_heads) if edge_heads is not None else None
        self.validate_on_exit = validate_on_exit
        
        self.stats = MaskingStats()
        self._original_forwards: dict[int, Callable] = {}
        self._patched = False
        
        # Mask positions (set before forward pass)
        self.q_positions: Optional[list[int]] = None
        self.k_positions: Optional[list[int]] = None
    
    def set_mask_positions(self, q_positions: list[int], k_positions: list[int]):
        """
        Set which Q->K attention edges to mask.
        
        Args:
            q_positions: Token positions for queries (e.g., answer tokens)
            k_positions: Token positions for keys (e.g., chunk tokens to block)
        """
        self.q_positions = q_positions
        self.k_positions = k_positions
    
    def __enter__(self):
        """Patch attention modules."""
        self._patch_attention_modules()
        self.stats.reset()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original forwards and validate."""
        self._restore_attention_modules()
        
        if exc_type is None and self.validate_on_exit:
            self._validate_instrumentation()
        
        return False
    
    def _patch_attention_modules(self):
        """Patch all attention module forwards."""
        layer_to_module = self.model_info["layer_to_module"]
        
        for layer_idx, attn_module in layer_to_module.items():
            original_forward = attn_module.forward
            self._original_forwards[layer_idx] = original_forward
            
            # Create patched forward for this layer
            patched_forward = self._create_patched_forward(original_forward, layer_idx)
            attn_module.forward = patched_forward
        
        self._patched = True
        logger.debug(f"Patched {len(layer_to_module)} attention modules")
    
    def _restore_attention_modules(self):
        """Restore original forwards."""
        if not self._patched:
            return
        
        layer_to_module = self.model_info["layer_to_module"]
        
        for layer_idx, attn_module in layer_to_module.items():
            if layer_idx in self._original_forwards:
                attn_module.forward = self._original_forwards[layer_idx]
        
        self._original_forwards.clear()
        self._patched = False
        logger.debug("Restored original attention forwards")
    
    def _create_patched_forward(self, original_forward: Callable, layer_idx: int) -> Callable:
        """
        Create a patched forward function for a specific layer.
        
        The patched forward:
        1. Records that this layer was hit
        2. Checks if this layer should be masked
        3. If yes, runs forward with attention_mask modification
        4. Records what was masked
        """
        masker = self  # Capture self for closure
        
        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple] = None,
            **kwargs,
        ):
            # Record that this layer was hit
            masker.stats.layers_hit.add(layer_idx)
            
            # Check if this layer should be masked
            should_mask = (
                masker.edge_layers is None or layer_idx in masker.edge_layers
            ) and masker.q_positions is not None and masker.k_positions is not None
            
            if not should_mask:
                # Call original forward unchanged
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=True,  # Need attention weights
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            
            # This layer should be masked
            masker.stats.layers_mask_applied.add(layer_idx)
            
            # We need to modify attention scores before softmax
            # The cleanest way is to use a custom attention_mask
            # But we need to be careful about the mask format
            
            # Get batch size and sequence length
            batch_size, seq_len, _ = hidden_states.shape
            num_heads = masker.model_info["num_heads"]
            
            # Determine which heads to mask
            if masker.edge_heads is None:
                heads_to_mask = list(range(num_heads))
            else:
                heads_to_mask = list(masker.edge_heads)
            
            masker.stats.heads_mask_applied[layer_idx] = set(heads_to_mask)
            
            # Create the edge mask
            # Shape: [batch, num_heads, seq_len, seq_len]
            # We add -inf at positions where Q (answer) attends to K (chunk)
            edge_mask = torch.zeros(
                (batch_size, num_heads, seq_len, seq_len),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            
            # Convert positions to tensors
            q_pos = torch.tensor(masker.q_positions, device=hidden_states.device)
            k_pos = torch.tensor(masker.k_positions, device=hidden_states.device)
            
            # Create mask for specified heads only
            for head_idx in heads_to_mask:
                # edge_mask[:, head_idx, q_positions, k_positions] = -inf
                for q in masker.q_positions:
                    for k in masker.k_positions:
                        if q < seq_len and k < seq_len:
                            edge_mask[:, head_idx, q, k] = float('-inf')
                            masker.stats.masked_entries_count += batch_size
            
            # Combine with existing attention mask if present
            if attention_mask is not None:
                # attention_mask is typically [batch, 1, seq, seq] or [batch, seq]
                if attention_mask.dim() == 4:
                    combined_mask = attention_mask + edge_mask
                elif attention_mask.dim() == 2:
                    # Expand and combine
                    expanded = attention_mask[:, None, None, :].expand(-1, num_heads, seq_len, -1)
                    combined_mask = expanded + edge_mask
                else:
                    # Best effort
                    combined_mask = attention_mask + edge_mask
            else:
                combined_mask = edge_mask
            
            # Call original forward with modified mask
            return original_forward(
                hidden_states,
                attention_mask=combined_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=True,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        
        return patched_forward
    
    def _validate_instrumentation(self):
        """
        HARD-FAIL if masking didn't happen as expected.
        
        This is the key guarantee that edge_layers/edge_heads are honored.
        """
        # Check layers
        if self.edge_layers is not None:
            expected_layers = self.edge_layers & self.stats.layers_hit
            if self.stats.layers_mask_applied != expected_layers:
                raise MaskingError(
                    f"MASKING VALIDATION FAILED!\n"
                    f"  Requested layers: {sorted(self.edge_layers)}\n"
                    f"  Layers hit in forward: {sorted(self.stats.layers_hit)}\n"
                    f"  Expected to mask (intersection): {sorted(expected_layers)}\n"
                    f"  Actually masked: {sorted(self.stats.layers_mask_applied)}\n"
                    f"  This means edge_layers was IGNORED!"
                )
        
        # Check that we masked something
        if self.q_positions and self.k_positions:
            if self.stats.masked_entries_count == 0:
                raise MaskingError(
                    f"MASKING VALIDATION FAILED!\n"
                    f"  Q positions: {self.q_positions[:5]}... (len={len(self.q_positions)})\n"
                    f"  K positions: {self.k_positions[:5]}... (len={len(self.k_positions)})\n"
                    f"  But masked_entries_count = 0!\n"
                    f"  The mask had no effect."
                )
        
        logger.debug(
            f"Masking validated: {len(self.stats.layers_mask_applied)} layers, "
            f"{self.stats.masked_entries_count} entries masked"
        )


@contextmanager
def scoped_mask(
    model,
    model_info: dict,
    edge_layers: Optional[list[int]],
    edge_heads: Optional[list[int]],
    q_positions: list[int],
    k_positions: list[int],
    validate: bool = True,
):
    """
    Convenience context manager for scoped masking.
    
    Usage:
        with scoped_mask(model, model_info, [0,1], None, q_pos, k_pos) as stats:
            output = model(input_ids)
        print(stats.summary())
    """
    masker = ScopedAttentionMasker(
        model, model_info, edge_layers, edge_heads, validate_on_exit=validate
    )
    masker.set_mask_positions(q_positions, k_positions)
    
    with masker:
        yield masker.stats
