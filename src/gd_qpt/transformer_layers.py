"""Transformer layers for QPT using JAX and Haiku"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, Tuple
import numpy as np


class MultiHeadAttention(hk.Module):
    """Multi-head attention module for transformer."""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        """Initialize multi-head attention.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            dropout_rate: Dropout rate
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray,
                 mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True,
                 return_attention: bool = False) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model)
            key: Key tensor of shape (batch, seq_len, d_model)
            value: Value tensor of shape (batch, seq_len, d_model)
            mask: Optional mask tensor
            is_training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Linear projections in batch from d_model => h x d_k
        Q = hk.Linear(self.d_model, with_bias=False)(query)
        K = hk.Linear(self.d_model, with_bias=False)(key)
        V = hk.Linear(self.d_model, with_bias=False)(value)

        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention scores
        scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            mask = jnp.expand_dims(mask, axis=1)  # Add head dimension
            scores = scores + (mask * -1e9)

        # Softmax
        attention_weights = jax.nn.softmax(scores, axis=-1)

        # Dropout
        if is_training:
            attention_weights = hk.dropout(hk.next_rng_key(), self.dropout_rate, attention_weights)

        # Apply attention to values
        context = jnp.matmul(attention_weights, V)

        # Reshape back to (batch, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Final linear projection
        output = hk.Linear(self.d_model)(context)

        if return_attention:
            return output, attention_weights
        return output, None


class TransformerEncoderLayer(hk.Module):
    """Single transformer encoder layer."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout_rate: float = 0.1):
        """Initialize transformer encoder layer.

        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True,
                 return_attention: bool = False) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Forward pass of transformer encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Output tensor and optionally attention weights
        """
        # Multi-head attention
        attn_module = MultiHeadAttention(self.d_model, self.num_heads, self.dropout_rate)
        attn_output, attention_weights = attn_module(x, x, x, mask, is_training, return_attention)

        # Dropout and residual connection
        if is_training:
            attn_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn_output)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + attn_output)

        # Feed-forward network
        ff_output = hk.Linear(self.d_ff)(x)
        ff_output = jax.nn.relu(ff_output)
        ff_output = hk.Linear(self.d_model)(ff_output)

        # Dropout and residual connection
        if is_training:
            ff_output = hk.dropout(hk.next_rng_key(), self.dropout_rate, ff_output)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x + ff_output)

        return x, attention_weights


class TransformerEncoder(hk.Module):
    """Stack of transformer encoder layers."""

    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout_rate: float = 0.1):
        """Initialize transformer encoder.

        Args:
            num_layers: Number of encoder layers
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward network
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None,
                 is_training: bool = True,
                 return_all_attention: bool = False) -> Tuple[jnp.ndarray, Optional[list]]:
        """Forward pass of transformer encoder.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_training: Whether in training mode
            return_all_attention: Whether to return all attention weights

        Returns:
            Output tensor and optionally list of attention weights from all layers
        """
        attention_weights_list = []

        for i in range(self.num_layers):
            layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.d_ff, self.dropout_rate
            )
            x, attn_weights = layer(x, mask, is_training, return_attention=return_all_attention)

            if return_all_attention and attn_weights is not None:
                attention_weights_list.append(attn_weights)

        if return_all_attention:
            return x, attention_weights_list
        return x, None


class MeasurementEncoder(hk.Module):
    """Encoder for measurement data in QPT."""

    def __init__(self, N: int, d_model: int):
        """Initialize measurement encoder.

        Args:
            N: Hilbert space dimension (2^n_qubits)
            d_model: Output dimension
        """
        super().__init__()
        self.N = N
        self.d_model = d_model

    def __call__(self, measurements: jnp.ndarray) -> jnp.ndarray:
        """Encode measurement data.

        Args:
            measurements: Measurement data of shape (batch, n_measurements, 2*N*N)
                         where 2*N*N accounts for real and imaginary parts

        Returns:
            Encoded measurements of shape (batch, n_measurements, d_model)
        """
        batch_size, n_measurements, _ = measurements.shape

        # First linear projection
        x = hk.Linear(self.d_model)(measurements)
        x = jax.nn.gelu(x)

        # Second projection with residual
        x_proj = hk.Linear(self.d_model)(x)
        x = x + x_proj

        # Layer normalization
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)

        return x