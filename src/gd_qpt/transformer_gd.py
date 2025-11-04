"""Transformer-enhanced Gradient Descent QPT combining attention with Stiefel optimization"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
from functools import partial
from typing import Optional, Dict, Tuple
from tqdm.auto import tqdm
from qutip import rand_unitary

from gd_qpt.gd import GradientDescent, stiefel_update, get_block, get_unblock, loss
from gd_qpt.transformer_qpt import TransformerQPT, AdaptiveMeasurementSelector
from gd_qpt.core import choi


class TransformerGradientDescent(GradientDescent):
    """Gradient descent QPT enhanced with transformer initialization and adaptive sampling."""

    def __init__(self, N: int, num_kraus: int, lr: float = 0.1, alpha: float = 0.999,
                 transformer_config: Optional[Dict] = None):
        """Initialize Transformer-enhanced gradient descent.

        Args:
            N: Hilbert space dimension
            num_kraus: Number of Kraus operators
            lr: Initial learning rate
            alpha: Learning rate decay factor
            transformer_config: Configuration for transformer model
        """
        super().__init__(N, num_kraus, lr, alpha)

        # Default transformer configuration
        self.transformer_config = transformer_config or {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 1024,
            'dropout_rate': 0.1
        }

        # Initialize transformer model parameters
        self.transformer_params = None
        self.transformer_state = None

        # For adaptive measurement selection
        self.use_adaptive = True
        self.measurement_selector = None

    def initialize_transformer(self, sample_data: jnp.ndarray,
                             sample_probes: jnp.ndarray,
                             sample_measurements: jnp.ndarray):
        """Initialize the transformer model with sample data.

        Args:
            sample_data: Sample measurement data
            sample_probes: Sample probe operators
            sample_measurements: Sample measurement operators
        """
        # Create the transformer model using Haiku
        def transformer_fn(data, probes, measurements):
            model = TransformerQPT(
                self.N, self.num_kraus, **self.transformer_config
            )
            return model(data, probes, measurements, is_training=True)

        # Transform to pure functions
        transformer = hk.without_apply_rng(hk.transform(transformer_fn))

        # Initialize parameters
        rng = jax.random.PRNGKey(42)
        self.transformer_params = transformer.init(
            rng, sample_data, sample_probes, sample_measurements
        )

        # Create apply function
        self.transformer_apply = transformer.apply

        # Initialize optimizer for transformer
        self.transformer_optimizer = optax.adam(1e-4)
        self.transformer_state = self.transformer_optimizer.init(self.transformer_params)

        # Create measurement selector
        self.measurement_selector = AdaptiveMeasurementSelector(self)

    def pretrain_transformer(self, data: jnp.ndarray, probes: jnp.ndarray,
                           measurements: jnp.ndarray, n_epochs: int = 50,
                           batch_size: int = 32):
        """Pre-train the transformer model before gradient descent.

        Args:
            data: Training data
            probes: Probe operators
            measurements: Measurement operators
            n_epochs: Number of pre-training epochs
            batch_size: Batch size for pre-training
        """
        if self.transformer_params is None:
            self.initialize_transformer(data, probes, measurements)

        @jax.jit
        def transformer_loss(params, data_batch, probes_batch, measurements_batch):
            """Compute transformer pre-training loss."""
            result = self.transformer_apply(
                params, data_batch, probes_batch, measurements_batch
            )
            kraus_pred = result['kraus']

            # Convert to predictions
            from gd_qpt.gd import predict
            data_pred = predict(kraus_pred, probes_batch, measurements_batch)

            # MSE loss
            mse_loss = jnp.mean((data_batch - data_pred) ** 2)

            # Regularization for CPTP constraint
            kraus_block = get_block(kraus_pred)
            cptp_loss = jnp.mean((kraus_block.T.conj() @ kraus_block - jnp.eye(self.N)) ** 2)

            return mse_loss + 0.1 * cptp_loss

        @jax.jit
        def update_step(params, opt_state, data_batch, probes_batch, measurements_batch):
            """Single update step for transformer."""
            loss_val, grads = jax.value_and_grad(transformer_loss)(
                params, data_batch, probes_batch, measurements_batch
            )
            updates, opt_state = self.transformer_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        print("Pre-training transformer...")
        for epoch in tqdm(range(n_epochs)):
            # Generate random batch
            n_data = data.shape[0]
            batch_idx = np.random.choice(n_data, min(batch_size, n_data), replace=False)

            # Sample corresponding probes and measurements
            n_probes = probes.shape[0]
            n_measurements = measurements.shape[0]
            probe_idx = np.random.choice(n_probes, min(batch_size, n_probes), replace=False)
            meas_idx = np.random.choice(n_measurements, min(batch_size, n_measurements), replace=False)

            # Create batch
            data_batch = data[batch_idx]
            probes_batch = probes[probe_idx]
            measurements_batch = measurements[meas_idx]

            # Update
            self.transformer_params, self.transformer_state, loss_val = update_step(
                self.transformer_params, self.transformer_state,
                data_batch, probes_batch, measurements_batch
            )

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val:.6f}")

    def fit(self, data: jnp.ndarray, probes: jnp.ndarray,
            measurements: jnp.ndarray, batch_size: int,
            maxiters: int = 1000, use_transformer: bool = True,
            pretrain_epochs: int = 50):
        """Fit the QPT model using transformer-enhanced gradient descent.

        Args:
            data: Measurement data
            probes: Probe operators
            measurements: Measurement operators
            batch_size: Batch size for gradient descent
            maxiters: Maximum iterations
            use_transformer: Whether to use transformer enhancement
            pretrain_epochs: Number of pre-training epochs for transformer

        Returns:
            Reconstructed Choi matrix
        """
        if use_transformer:
            # Pre-train transformer if not already done
            if self.transformer_params is None:
                self.pretrain_transformer(
                    data, probes, measurements, pretrain_epochs, batch_size
                )

            # Get transformer initialization
            print("Getting transformer initialization...")
            result = self.transformer_apply(
                self.transformer_params, data, probes, measurements
            )
            kraus_init = result['kraus']
            uncertainty = result['uncertainty']

            # Use transformer initialization
            params = get_block(kraus_init)

            # Adapt learning rate based on uncertainty
            lr = self.lr / (1 + jnp.mean(uncertainty))
            print(f"Adaptive learning rate: {lr:.6f} (based on uncertainty)")
        else:
            # Standard random initialization
            params_init = jnp.array([
                rand_unitary(self.N, density=0.5).full() / np.sqrt(self.num_kraus)
                for _ in range(self.num_kraus)
            ])
            params = get_block(params_init)
            lr = self.lr
            uncertainty = None

        # Gradient descent with optional adaptive sampling
        print("Starting gradient descent optimization...")
        for step in tqdm(range(maxiters)):
            if use_transformer and self.use_adaptive and step > 0:
                # Adaptive batch selection using transformer attention
                current_kraus = get_unblock(params, self.num_kraus)

                # Compute current predictions
                from gd_qpt.gd import predict
                current_pred = predict(current_kraus, probes, measurements)

                # Get informative measurements
                n_total = probes.shape[0] * measurements.shape[0]
                candidate_indices = jnp.arange(n_total)

                # Select batch of most informative measurements
                selected_indices = []
                for _ in range(batch_size):
                    idx = self.measurement_selector.select_next_measurement(
                        current_pred, probes, measurements, candidate_indices
                    )
                    selected_indices.append(idx)

                # Convert to probe and measurement indices
                selected_indices = jnp.array(selected_indices)
                idx1 = selected_indices // measurements.shape[0]
                idx2 = selected_indices % measurements.shape[0]

                # Create corresponding data indices
                idx = tuple(np.meshgrid(idx1, idx2))
            else:
                # Standard random batch selection
                from gd_qpt.gd import generate_batch
                idx, idx1, idx2 = generate_batch(batch_size, probes.shape[0], measurements.shape[0])

            # Compute gradients
            grads = jax.grad(loss)(params, data.T[idx].real, probes[idx1],
                                  measurements[idx2], num_kraus=self.num_kraus)
            grads = jnp.conj(grads)

            # Normalize gradients
            grad_norm = jnp.linalg.norm(grads)
            grads = grads / (grad_norm + 1e-8)

            # Apply uncertainty weighting if available
            if uncertainty is not None and use_transformer:
                # Weight gradient by inverse uncertainty
                confidence = 1.0 / (1.0 + jnp.mean(uncertainty))
                grads = grads * confidence

            # Stiefel manifold update
            params = stiefel_update(params, grads, lr)

            # Learning rate decay
            lr = self.alpha * lr

            # Optionally update transformer every few steps
            if use_transformer and step % 10 == 0 and step > 0:
                # Fine-tune transformer with current results
                current_kraus = get_unblock(params, self.num_kraus)

                @jax.jit
                def finetune_loss(trans_params):
                    result = self.transformer_apply(trans_params, data, probes, measurements)
                    trans_kraus = result['kraus']
                    return jnp.mean((trans_kraus - current_kraus) ** 2)

                trans_grads = jax.grad(finetune_loss)(self.transformer_params)
                updates, self.transformer_state = self.transformer_optimizer.update(
                    trans_grads, self.transformer_state
                )
                self.transformer_params = optax.apply_updates(
                    self.transformer_params, updates
                )

                # Update uncertainty estimate
                result = self.transformer_apply(
                    self.transformer_params, data, probes, measurements
                )
                uncertainty = result['uncertainty']

        # Final Kraus operators
        k_ops = get_unblock(params, self.num_kraus)

        # Compute Choi matrix
        choi_gd_pred = choi(k_ops)

        return choi_gd_pred


def create_and_train_model(N: int, num_kraus: int, data: jnp.ndarray,
                          probes: jnp.ndarray, measurements: jnp.ndarray,
                          batch_size: int = 32, maxiters: int = 1000,
                          use_transformer: bool = True) -> jnp.ndarray:
    """Convenience function to create and train TransformerGradientDescent model.

    Args:
        N: Hilbert space dimension
        num_kraus: Number of Kraus operators
        data: Measurement data
        probes: Probe operators
        measurements: Measurement operators
        batch_size: Batch size
        maxiters: Maximum iterations
        use_transformer: Whether to use transformer enhancement

    Returns:
        Reconstructed Choi matrix
    """
    model = TransformerGradientDescent(N, num_kraus)
    choi_matrix = model.fit(
        data, probes, measurements, batch_size,
        maxiters=maxiters, use_transformer=use_transformer
    )
    return choi_matrix