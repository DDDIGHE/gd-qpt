"""Utility functions for Transformer-based QPT"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import haiku as hk

class QuantumPositionalEncoding(hk.Module):
    """Positional encoding that respects quantum tensor product structure."""

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def __call__(self, seq_len: int, n_qubits: int = None):
        """Generate positional encoding.

        Args:
            seq_len: Length of the sequence
            n_qubits: Number of qubits (for quantum structure awareness)

        Returns:
            Positional encoding of shape (seq_len, d_model)
        """
        # Standard sinusoidal encoding
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) *
                          -(jnp.log(10000.0) / self.d_model))

        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

        # Add quantum structure if n_qubits is specified
        if n_qubits is not None:
            # Encode qubit indices in additional dimensions
            qubit_encoding = self._quantum_structure_encoding(seq_len, n_qubits)
            if qubit_encoding.shape[1] <= self.d_model:
                pe = pe.at[:, :qubit_encoding.shape[1]].add(0.1 * qubit_encoding)

        return pe

    def _quantum_structure_encoding(self, seq_len: int, n_qubits: int):
        """Encode the tensor product structure of quantum states.

        Args:
            seq_len: Sequence length (should be 6^n_qubits for Pauli measurements)
            n_qubits: Number of qubits

        Returns:
            Quantum structure encoding
        """
        # For Pauli measurements, we have 6 basis states per qubit
        n_basis = 6

        # Create encoding based on qubit structure
        encoding = []
        for idx in range(seq_len):
            # Decompose index into qubit-wise basis indices
            qubit_indices = []
            temp_idx = idx
            for _ in range(n_qubits):
                qubit_indices.append(temp_idx % n_basis)
                temp_idx //= n_basis

            # Encode as a vector
            enc_vec = jnp.array(qubit_indices) / n_basis
            encoding.append(enc_vec)

        encoding = jnp.array(encoding)

        # Pad or truncate to match model dimension
        if encoding.shape[1] < self.d_model:
            padding = jnp.zeros((seq_len, self.d_model - encoding.shape[1]))
            encoding = jnp.concatenate([encoding, padding], axis=1)
        else:
            encoding = encoding[:, :self.d_model]

        return encoding


def create_attention_mask(batch_size: int, seq_len: int,
                         causal: bool = False) -> jnp.ndarray:
    """Create attention mask for transformer.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        causal: Whether to use causal masking

    Returns:
        Attention mask of shape (batch_size, seq_len, seq_len)
    """
    if causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = jnp.expand_dims(mask, 0)
        mask = jnp.repeat(mask, batch_size, axis=0)
    else:
        mask = jnp.ones((batch_size, seq_len, seq_len))

    return mask


def compute_attention_entropy(attention_weights: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy of attention weights for uncertainty estimation.

    Args:
        attention_weights: Attention weights of shape (batch, heads, seq_len, seq_len)

    Returns:
        Entropy values of shape (batch,)
    """
    # Average over heads
    avg_attention = jnp.mean(attention_weights, axis=1)

    # Compute entropy
    entropy = -jnp.sum(avg_attention * jnp.log(avg_attention + 1e-10), axis=-1)

    # Average over sequence
    entropy = jnp.mean(entropy, axis=-1)

    return entropy