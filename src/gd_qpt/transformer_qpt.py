"""Transformer-based Quantum Process Tomography with Stiefel manifold optimization"""

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional, Tuple, Dict
from functools import partial

from gd_qpt.gd import stiefel_update, get_block, get_unblock
from gd_qpt.transformer_layers import TransformerEncoder, MeasurementEncoder
from gd_qpt.transformer_utils import QuantumPositionalEncoding, compute_attention_entropy


class TransformerQPT(hk.Module):
    """Transformer-based QPT with attention mechanisms and Stiefel projection."""

    def __init__(self, N: int, num_kraus: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 1024,
                 dropout_rate: float = 0.1):
        """Initialize TransformerQPT.

        Args:
            N: Hilbert space dimension (2^n_qubits)
            num_kraus: Number of Kraus operators
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.N = N
        self.num_kraus = num_kraus
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        # Calculate number of qubits
        self.n_qubits = int(np.log2(N))

    def __call__(self, data: jnp.ndarray, probes: jnp.ndarray,
                 measurements: jnp.ndarray, is_training: bool = True,
                 return_attention: bool = False) -> Dict:
        """Forward pass of TransformerQPT.

        Args:
            data: Measurement data of shape (n_measurements, n_probes)
            probes: Probe operators of shape (n_probes, N, N)
            measurements: Measurement operators of shape (n_measurements, N, N)
            is_training: Whether in training mode
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
                - 'kraus': Kraus operators (num_kraus, N, N)
                - 'uncertainty': Uncertainty estimates (num_kraus,)
                - 'attention': Optional attention weights
        """
        # Reshape and prepare data
        batch_size = 1  # Process as single batch
        n_measurements = measurements.shape[0]
        n_probes = probes.shape[0]

        # Flatten and concatenate measurement data with probe/measurement info
        # Shape: (batch, n_measurements * n_probes, feature_dim)
        data_flat = data.T.flatten()  # Flatten measurement data
        seq_len = n_measurements * n_probes

        # Create feature vectors for each measurement-probe pair
        features = self._create_features(data_flat, probes, measurements)

        # Add batch dimension
        features = jnp.expand_dims(features, 0)  # (1, seq_len, feature_dim)

        # Measurement encoding
        encoder = MeasurementEncoder(self.N, self.d_model)
        encoded = encoder(features)  # (1, seq_len, d_model)

        # Add positional encoding
        pos_encoder = QuantumPositionalEncoding(self.d_model)
        pos_encoding = pos_encoder(seq_len, self.n_qubits)
        encoded = encoded + pos_encoding

        # Apply transformer
        transformer = TransformerEncoder(
            self.n_layers, self.d_model, self.n_heads, self.d_ff, self.dropout_rate
        )
        transformed, attention_weights = transformer(
            encoded, is_training=is_training, return_all_attention=return_attention
        )

        # Aggregate sequence information
        # Use mean pooling over sequence dimension
        aggregated = jnp.mean(transformed, axis=1)  # (1, d_model)

        # Decode to Kraus operators
        kraus_flat = hk.Linear(2 * self.N * self.N * self.num_kraus)(aggregated)
        kraus_flat = kraus_flat.reshape(self.num_kraus, self.N, self.N, 2)

        # Convert to complex
        kraus = kraus_flat[..., 0] + 1j * kraus_flat[..., 1]  # (num_kraus, N, N)

        # Uncertainty estimation from attention entropy
        uncertainty = self._estimate_uncertainty(attention_weights, aggregated)

        # Project to Stiefel manifold
        kraus_block = get_block(kraus)
        kraus_stiefel = self._project_stiefel(kraus_block)
        kraus_final = get_unblock(kraus_stiefel, self.num_kraus)

        result = {
            'kraus': kraus_final,
            'uncertainty': uncertainty,
        }

        if return_attention:
            result['attention'] = attention_weights

        return result

    def _create_features(self, data: jnp.ndarray, probes: jnp.ndarray,
                         measurements: jnp.ndarray) -> jnp.ndarray:
        """Create feature vectors from measurement data and operators.

        Args:
            data: Flattened measurement data
            probes: Probe operators
            measurements: Measurement operators

        Returns:
            Feature vectors of shape (seq_len, feature_dim)
        """
        n_measurements = measurements.shape[0]
        n_probes = probes.shape[0]
        seq_len = n_measurements * n_probes

        # Reshape data back to (n_measurements, n_probes)
        data_2d = data.reshape(n_measurements, n_probes).T

        features_list = []

        for i in range(n_probes):
            for j in range(n_measurements):
                # Measurement value
                meas_val = jnp.array([data_2d[i, j]])

                # Probe and measurement operator features (flatten and take real/imag)
                probe_feat = jnp.concatenate([
                    probes[i].real.flatten(),
                    probes[i].imag.flatten()
                ])[:self.N * self.N]  # Truncate to reasonable size

                meas_feat = jnp.concatenate([
                    measurements[j].real.flatten(),
                    measurements[j].imag.flatten()
                ])[:self.N * self.N]

                # Concatenate all features
                feat_vec = jnp.concatenate([meas_val, probe_feat, meas_feat])
                features_list.append(feat_vec)

        features = jnp.array(features_list)

        # Ensure consistent feature dimension
        feature_dim = 2 * self.N * self.N * 2 + 1
        if features.shape[1] < feature_dim:
            padding = jnp.zeros((seq_len, feature_dim - features.shape[1]))
            features = jnp.concatenate([features, padding], axis=1)
        else:
            features = features[:, :feature_dim]

        return features

    def _project_stiefel(self, kraus_block: jnp.ndarray) -> jnp.ndarray:
        """Project Kraus operators onto Stiefel manifold for CPTP constraint.

        Args:
            kraus_block: Block matrix of Kraus operators

        Returns:
            Projected Kraus operators on Stiefel manifold
        """
        # Use QR decomposition for projection
        Q, R = jnp.linalg.qr(kraus_block)
        return Q

    def _estimate_uncertainty(self, attention_weights: Optional[list],
                             aggregated: jnp.ndarray) -> jnp.ndarray:
        """Estimate uncertainty from attention and aggregated features.

        Args:
            attention_weights: List of attention weights from each layer
            aggregated: Aggregated features

        Returns:
            Uncertainty estimates for each Kraus operator
        """
        # Uncertainty from attention entropy
        if attention_weights is not None and len(attention_weights) > 0:
            # Use last layer attention
            last_attention = attention_weights[-1]
            entropy = compute_attention_entropy(last_attention)
        else:
            entropy = jnp.zeros(1)

        # Additional uncertainty head
        uncertainty_logits = hk.Sequential([
            hk.Linear(self.d_model // 2),
            jax.nn.relu,
            hk.Linear(self.num_kraus),
            jax.nn.softplus  # Ensure positive uncertainty
        ])(aggregated)

        # Combine entropy and learned uncertainty
        uncertainty = uncertainty_logits.squeeze() * (1 + entropy.mean())

        return uncertainty


class AdaptiveMeasurementSelector:
    """Select next measurements based on attention-derived information gain."""

    def __init__(self, transformer_model: TransformerQPT):
        """Initialize selector with transformer model.

        Args:
            transformer_model: Trained TransformerQPT model
        """
        self.model = transformer_model

    def select_next_measurement(self, current_data: jnp.ndarray,
                               probes: jnp.ndarray,
                               measurements: jnp.ndarray,
                               candidate_indices: jnp.ndarray) -> int:
        """Select the next most informative measurement.

        Args:
            current_data: Current measurement data
            probes: Available probe states
            measurements: Available measurement operators
            candidate_indices: Indices of candidate measurements to consider

        Returns:
            Index of the most informative measurement
        """
        # Get model predictions with attention
        result = self.model(current_data, probes, measurements,
                           is_training=False, return_attention=True)

        attention_weights = result.get('attention', None)
        if attention_weights is None or len(attention_weights) == 0:
            # Random selection if no attention available
            return np.random.choice(candidate_indices)

        # Compute information gain from attention entropy
        last_attention = attention_weights[-1]  # Shape: (batch, heads, seq, seq)

        # Average over batch and heads
        avg_attention = jnp.mean(last_attention, axis=(0, 1))  # (seq, seq)

        # Compute entropy for each position
        entropy = -jnp.sum(avg_attention * jnp.log(avg_attention + 1e-10), axis=-1)

        # Map sequence position to measurement indices
        n_measurements = measurements.shape[0]
        n_probes = probes.shape[0]

        # Reshape entropy to (n_probes, n_measurements)
        entropy_2d = entropy.reshape(n_probes, n_measurements)

        # Compute information gain for candidate measurements
        info_gains = []
        for idx in candidate_indices:
            probe_idx = idx // n_measurements
            meas_idx = idx % n_measurements
            info_gain = entropy_2d[probe_idx, meas_idx]
            info_gains.append(info_gain)

        info_gains = jnp.array(info_gains)

        # Select measurement with highest information gain
        best_idx = candidate_indices[jnp.argmax(info_gains)]

        return best_idx