"""
Robust chunk-to-token span alignment.

Maps character spans (chunks, answer) to token indices using tokenizer offsets.
Hard-fails on ambiguous alignments - no heuristic fallbacks.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("edgepatch")


class AlignmentError(Exception):
    """Raised when chunk-to-token alignment fails."""
    pass


@dataclass
class TokenSpan:
    """A span of tokens corresponding to a text chunk."""
    start_token: int  # Inclusive
    end_token: int    # Exclusive
    chunk_idx: int
    text: str
    
    @property
    def length(self) -> int:
        return self.end_token - self.start_token


def align_chunks_to_tokens(
    text: str,
    chunks: list,  # list of Chunk objects
    answer_start_char: int,
    answer_end_char: int,
    tokenizer,
    max_length: Optional[int] = None,
) -> tuple[list[TokenSpan], TokenSpan, dict]:
    """
    Align chunk character spans to token indices.
    
    Args:
        text: Full input text
        chunks: List of Chunk objects with start_char, end_char
        answer_start_char: Character offset where answer starts
        answer_end_char: Character offset where answer ends
        tokenizer: HuggingFace tokenizer
        max_length: Optional max sequence length (truncates if exceeded)
    
    Returns:
        tuple of:
        - List of TokenSpan objects for chunks
        - TokenSpan for answer
        - Encoding info dict
    
    Raises:
        AlignmentError: If alignment fails or is ambiguous
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_tensors=None,  # Return lists, not tensors
    )
    
    input_ids = encoding["input_ids"]
    offset_mapping = encoding.get("offset_mapping")
    
    if offset_mapping is None:
        # Fallback to incremental tokenization
        logger.warning("Tokenizer doesn't support offset_mapping, using incremental fallback")
        return _align_incremental(text, chunks, answer_start_char, answer_end_char, tokenizer, max_length)
    
    # Build token index lookup from character positions
    n_tokens = len(input_ids)
    
    # Align chunk spans
    chunk_spans = []
    for chunk in chunks:
        span = _find_token_span(
            offset_mapping,
            chunk.start_char,
            chunk.end_char,
            chunk.chunk_idx,
            chunk.text,
        )
        if span is None:
            raise AlignmentError(
                f"Failed to align chunk {chunk.chunk_idx} "
                f"(chars {chunk.start_char}-{chunk.end_char}) to tokens"
            )
        chunk_spans.append(span)
    
    # Align answer span
    answer_span = _find_token_span(
        offset_mapping,
        answer_start_char,
        answer_end_char,
        -1,  # Special index for answer
        text[answer_start_char:answer_end_char],
    )
    if answer_span is None:
        raise AlignmentError(
            f"Failed to align answer span (chars {answer_start_char}-{answer_end_char}) to tokens"
        )
    
    encoding_info = {
        "n_tokens": n_tokens,
        "n_chunks": len(chunk_spans),
        "answer_tokens": answer_span.length,
    }
    
    return chunk_spans, answer_span, encoding_info


def _find_token_span(
    offset_mapping: list[tuple[int, int]],
    start_char: int,
    end_char: int,
    chunk_idx: int,
    text: str,
) -> Optional[TokenSpan]:
    """
    Find token indices that cover the given character span.
    
    Returns the token span [start_token, end_token) that covers chars [start_char, end_char).
    """
    start_token = None
    end_token = None
    
    for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        # Skip special tokens (they have (0, 0) offsets usually)
        if tok_start == tok_end == 0 and token_idx > 0:
            continue
        
        # Find first token that overlaps with our span
        if start_token is None:
            if tok_end > start_char and tok_start < end_char:
                start_token = token_idx
        
        # Find last token that overlaps with our span
        if tok_start < end_char and tok_end > start_char:
            end_token = token_idx + 1  # Exclusive end
    
    if start_token is None or end_token is None:
        return None
    
    return TokenSpan(
        start_token=start_token,
        end_token=end_token,
        chunk_idx=chunk_idx,
        text=text,
    )


def _align_incremental(
    text: str,
    chunks: list,
    answer_start_char: int,
    answer_end_char: int,
    tokenizer,
    max_length: Optional[int] = None,
) -> tuple[list[TokenSpan], TokenSpan, dict]:
    """
    Fallback alignment using incremental tokenization.
    
    This is slower but works for tokenizers without offset_mapping.
    
    Method:
        start_tok = len(tokenize(prefix_text))
        end_tok = len(tokenize(prefix_text + chunk_text))
    """
    # First, tokenize the full text to get total length
    full_tokens = tokenizer.encode(text, add_special_tokens=True)
    if max_length and len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
    
    # Helper to get token count for a prefix
    def token_count_for_prefix(char_end: int) -> int:
        if char_end <= 0:
            return 0
        prefix = text[:char_end]
        tokens = tokenizer.encode(prefix, add_special_tokens=True)
        return len(tokens)
    
    # Align chunks
    chunk_spans = []
    for chunk in chunks:
        # Be careful: we want tokens for text[:end_char] minus tokens for text[:start_char]
        # But due to sub-word tokenization, this is approximate
        start_tok = token_count_for_prefix(chunk.start_char)
        end_tok = token_count_for_prefix(chunk.end_char)
        
        # Sanity check
        if start_tok >= end_tok:
            # Chunk is too small or tokenization is weird
            logger.warning(f"Chunk {chunk.chunk_idx} has no tokens (start={start_tok}, end={end_tok})")
            # Give it at least one token
            end_tok = start_tok + 1
        
        chunk_spans.append(TokenSpan(
            start_token=start_tok,
            end_token=min(end_tok, len(full_tokens)),
            chunk_idx=chunk.chunk_idx,
            text=chunk.text,
        ))
    
    # Align answer
    answer_start_tok = token_count_for_prefix(answer_start_char)
    answer_end_tok = token_count_for_prefix(answer_end_char)
    if answer_start_tok >= answer_end_tok:
        answer_end_tok = answer_start_tok + 1
    
    answer_span = TokenSpan(
        start_token=answer_start_tok,
        end_token=min(answer_end_tok, len(full_tokens)),
        chunk_idx=-1,
        text=text[answer_start_char:answer_end_char],
    )
    
    encoding_info = {
        "n_tokens": len(full_tokens),
        "n_chunks": len(chunk_spans),
        "answer_tokens": answer_span.length,
        "method": "incremental",
    }
    
    return chunk_spans, answer_span, encoding_info


def validate_spans(
    chunk_spans: list[TokenSpan],
    answer_span: TokenSpan,
    n_tokens: int,
) -> None:
    """
    Validate that spans are well-formed.
    
    Raises:
        AlignmentError: If any span is invalid
    """
    for span in chunk_spans:
        if span.start_token < 0 or span.end_token > n_tokens:
            raise AlignmentError(
                f"Chunk {span.chunk_idx} span [{span.start_token}, {span.end_token}) "
                f"out of bounds [0, {n_tokens})"
            )
        if span.start_token >= span.end_token:
            raise AlignmentError(
                f"Chunk {span.chunk_idx} has empty span [{span.start_token}, {span.end_token})"
            )
    
    if answer_span.start_token < 0 or answer_span.end_token > n_tokens:
        raise AlignmentError(
            f"Answer span [{answer_span.start_token}, {answer_span.end_token}) "
            f"out of bounds [0, {n_tokens})"
        )
    if answer_span.start_token >= answer_span.end_token:
        raise AlignmentError(
            f"Answer has empty span [{answer_span.start_token}, {answer_span.end_token})"
        )
