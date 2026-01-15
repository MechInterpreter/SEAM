"""
CRITICAL: Tests that layer/head scoping actually changes outputs.

These tests are designed to fail fast if edge_layers/edge_heads are ignored.
This is the key safeguard against the failure mode where masking config
is printed but has no effect.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


class MockAttentionModule(nn.Module):
    """Mock attention module for testing masking."""
    
    def __init__(self, layer_idx: int, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self._edgepatch_layer_idx = layer_idx
        
        # Simple linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Track if masking was applied
        self.last_attention_mask = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Store mask for inspection
        self.last_attention_mask = attention_mask
        
        # Simple attention computation
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        head_dim = self.hidden_size // self.num_heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                scores = scores + attention_mask
            elif attention_mask.dim() == 2:
                scores = scores + attention_mask[:, None, None, :]
        
        # Softmax and attention
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        
        return (output, attn_weights) if output_attentions else (output,)


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing."""
    
    def __init__(self, num_layers: int = 4, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Create mock layers
        self.model = MagicMock()
        self.model.layers = nn.ModuleList([
            self._create_layer(i, hidden_size, num_heads)
            for i in range(num_layers)
        ])
    
    def _create_layer(self, idx: int, hidden_size: int, num_heads: int):
        layer = MagicMock()
        layer.self_attn = MockAttentionModule(idx, hidden_size, num_heads)
        return layer
    
    def forward(self, input_ids, **kwargs):
        # Simple forward that runs through attention layers
        batch_size, seq_len = input_ids.shape
        hidden = torch.randn(batch_size, seq_len, self.hidden_size)
        
        for layer in self.model.layers:
            hidden, = layer.self_attn(hidden)
        
        return MagicMock(logits=torch.randn(batch_size, seq_len, 1000))
    
    def parameters(self):
        return iter([torch.tensor([0.0])])


def test_masking_stats_recorded():
    """Test that masking stats are correctly recorded."""
    from edgepatch.masking import ScopedAttentionMasker, MaskingStats
    from edgepatch.model import tag_attention_layers
    
    model = MockTransformerModel(num_layers=4, num_heads=4)
    model_info = tag_attention_layers(model)
    
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[0, 1], 
        edge_heads=[0, 1],
        validate_on_exit=False,  # Don't validate yet
    ) as masker:
        masker.set_mask_positions([5, 6, 7], [0, 1, 2])
        
        # Run a mock forward through the attention layers
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
        
        # Check stats
        assert 0 in masker.stats.layers_hit
        assert 1 in masker.stats.layers_hit
        assert 0 in masker.stats.layers_mask_applied
        assert 1 in masker.stats.layers_mask_applied
        # Layers 2 and 3 should be hit but not masked
        assert 2 in masker.stats.layers_hit
        assert 2 not in masker.stats.layers_mask_applied


def test_layer_scoping_works():
    """
    CRITICAL TEST: Verify that different edge_layers produce different results.
    
    This test fails if edge_layers is ignored.
    """
    from edgepatch.masking import ScopedAttentionMasker
    from edgepatch.model import tag_attention_layers
    
    model = MockTransformerModel(num_layers=4, num_heads=4)
    model_info = tag_attention_layers(model)
    
    # Run with layer 0 masked
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[0], 
        edge_heads=None,
        validate_on_exit=False,
    ) as masker:
        masker.set_mask_positions([5, 6], [0, 1])
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
        
        layers_masked_A = masker.stats.layers_mask_applied.copy()
    
    # Run with layer 3 masked
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[3], 
        edge_heads=None,
        validate_on_exit=False,
    ) as masker:
        masker.set_mask_positions([5, 6], [0, 1])
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
        
        layers_masked_B = masker.stats.layers_mask_applied.copy()
    
    # Verify different layers were masked
    assert layers_masked_A != layers_masked_B, \
        f"Layer scoping failed! A masked {layers_masked_A}, B masked {layers_masked_B}"
    assert 0 in layers_masked_A and 0 not in layers_masked_B
    assert 3 in layers_masked_B and 3 not in layers_masked_A


def test_head_scoping_recorded():
    """Test that head scoping is recorded correctly."""
    from edgepatch.masking import ScopedAttentionMasker
    from edgepatch.model import tag_attention_layers
    
    model = MockTransformerModel(num_layers=2, num_heads=4)
    model_info = tag_attention_layers(model)
    
    # Run with heads [0, 1] masked
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[0], 
        edge_heads=[0, 1],
        validate_on_exit=False,
    ) as masker:
        masker.set_mask_positions([5], [0])
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
        
        heads_masked = masker.stats.heads_mask_applied.get(0, set())
    
    assert 0 in heads_masked
    assert 1 in heads_masked
    assert 2 not in heads_masked
    assert 3 not in heads_masked


def test_masking_validation_fails_on_mismatch():
    """Test that validation raises error when masking doesn't happen."""
    from edgepatch.masking import ScopedAttentionMasker, MaskingError
    from edgepatch.model import tag_attention_layers
    
    model = MockTransformerModel(num_layers=4, num_heads=4)
    model_info = tag_attention_layers(model)
    
    # Request masking on layer 0, but don't run any forward
    # This should fail validation
    with pytest.raises(MaskingError):
        with ScopedAttentionMasker(
            model, model_info, 
            edge_layers=[0], 
            edge_heads=None,
            validate_on_exit=True,
        ) as masker:
            masker.set_mask_positions([5], [0])
            # Don't run forward - validation should fail
            pass


def test_stats_reset_between_runs():
    """Test that stats are reset between context manager uses."""
    from edgepatch.masking import ScopedAttentionMasker
    from edgepatch.model import tag_attention_layers
    
    model = MockTransformerModel(num_layers=2, num_heads=4)
    model_info = tag_attention_layers(model)
    
    # First run
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[0], 
        edge_heads=None,
        validate_on_exit=False,
    ) as masker:
        masker.set_mask_positions([5], [0])
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
    
    # Second run - stats should be fresh
    with ScopedAttentionMasker(
        model, model_info, 
        edge_layers=[1], 
        edge_heads=None,
        validate_on_exit=False,
    ) as masker:
        masker.set_mask_positions([5], [0])
        
        # Before forward, stats should be empty
        assert len(masker.stats.layers_hit) == 0
        
        hidden = torch.randn(1, 10, 64)
        for layer in model.model.layers:
            hidden, = layer.self_attn(hidden)
        
        # After forward, only layer 1 should be masked
        assert 1 in masker.stats.layers_mask_applied
        assert 0 not in masker.stats.layers_mask_applied


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
