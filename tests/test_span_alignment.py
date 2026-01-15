"""
Tests for chunk-to-token span alignment.
"""

import pytest


class MockChunk:
    """Mock chunk object for testing."""
    def __init__(self, text: str, start_char: int, end_char: int, chunk_idx: int = 0):
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.chunk_idx = chunk_idx


class MockTokenizer:
    """Mock tokenizer for testing without loading a real model."""
    
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
    
    def __call__(self, text, return_offsets_mapping=False, truncation=False, 
                 max_length=None, return_tensors=None):
        # Simple word-based tokenization for testing
        words = text.split()
        input_ids = list(range(len(words)))
        
        result = {"input_ids": input_ids}
        
        if return_offsets_mapping:
            # Build offset mapping
            offsets = []
            pos = 0
            for word in words:
                start = text.find(word, pos)
                end = start + len(word)
                offsets.append((start, end))
                pos = end
            result["offset_mapping"] = offsets
        
        return result
    
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def test_basic_alignment():
    """Test basic chunk alignment with mock tokenizer."""
    from edgepatch.spans import align_chunks_to_tokens, TokenSpan
    
    text = "The quick brown fox jumps over the lazy dog"
    # Chunk: "brown fox" (chars 10-19)
    chunks = [MockChunk("brown fox", 10, 19, 0)]
    
    tokenizer = MockTokenizer()
    
    chunk_spans, answer_span, info = align_chunks_to_tokens(
        text, chunks, 35, 44,  # "lazy dog" as answer
        tokenizer
    )
    
    assert len(chunk_spans) == 1
    assert chunk_spans[0].chunk_idx == 0
    assert info["n_chunks"] == 1


def test_multiple_chunks():
    """Test alignment with multiple chunks."""
    from edgepatch.spans import align_chunks_to_tokens
    
    text = "One Two Three Four Five Six Seven Eight"
    chunks = [
        MockChunk("Two", 4, 7, 0),
        MockChunk("Four Five", 14, 23, 1),
    ]
    
    tokenizer = MockTokenizer()
    
    chunk_spans, answer_span, info = align_chunks_to_tokens(
        text, chunks, 28, 38,  # "Seven Eight" as answer
        tokenizer
    )
    
    assert len(chunk_spans) == 2
    assert info["n_chunks"] == 2


def test_validate_spans():
    """Test span validation."""
    from edgepatch.spans import validate_spans, TokenSpan, AlignmentError
    
    # Valid spans
    spans = [TokenSpan(0, 5, 0, "test")]
    answer = TokenSpan(10, 15, -1, "answer")
    validate_spans(spans, answer, 20)  # Should not raise
    
    # Invalid: out of bounds
    spans = [TokenSpan(0, 25, 0, "test")]
    with pytest.raises(AlignmentError):
        validate_spans(spans, answer, 20)
    
    # Invalid: empty span
    spans = [TokenSpan(5, 5, 0, "test")]
    with pytest.raises(AlignmentError):
        validate_spans(spans, answer, 20)


def test_token_span_properties():
    """Test TokenSpan dataclass properties."""
    from edgepatch.spans import TokenSpan
    
    span = TokenSpan(10, 20, 5, "test text")
    
    assert span.start_token == 10
    assert span.end_token == 20
    assert span.length == 10
    assert span.chunk_idx == 5
    assert span.text == "test text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
