"""Evaluation metrics and visualization tools for Transformer QPT"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from typing import Optional, Dict, List, Tuple


def process_fidelity(choi_true: jnp.ndarray, choi_pred: jnp.ndarray) -> float:
    """Compute process fidelity between true and predicted Choi matrices.

    Args:
        choi_true: True Choi matrix
        choi_pred: Predicted Choi matrix

    Returns:
        Process fidelity value between 0 and 1
    """
    # Normalize Choi matrices
    choi_true = choi_true / jnp.trace(choi_true)
    choi_pred = choi_pred / jnp.trace(choi_pred)

    # Compute fidelity using the formula: F = (Tr[sqrt(sqrt(ρ)σsqrt(ρ))])^2
    sqrt_true = jnp.linalg.sqrtm(choi_true)
    inner = sqrt_true @ choi_pred @ sqrt_true
    sqrt_inner = jnp.linalg.sqrtm(inner)
    fidelity = jnp.real(jnp.trace(sqrt_inner) ** 2)

    return float(fidelity)


def diamond_norm_distance(choi_true: jnp.ndarray, choi_pred: jnp.ndarray,
                          use_cvxpy: bool = False) -> float:
    """Compute diamond norm distance between two quantum processes.

    Args:
        choi_true: True Choi matrix
        choi_pred: Predicted Choi matrix
        use_cvxpy: Whether to use CVXPY for exact computation (requires installation)

    Returns:
        Diamond norm distance
    """
    if use_cvxpy:
        try:
            import cvxpy as cp
            # Implement exact diamond norm using SDP
            n = choi_true.shape[0] // 2

            # Variables
            J = cp.Variable((2*n, 2*n), hermitian=True)
            rho = cp.Variable((n, n), hermitian=True)

            # Constraints
            constraints = [
                cp.bmat([[rho, np.zeros((n, n))],
                        [np.zeros((n, n)), J]]) >> 0,
                cp.trace(rho) == 1,
                rho >> 0
            ]

            # Objective: maximize trace(J @ (choi_pred - choi_true))
            obj = cp.Maximize(cp.real(cp.trace(J @ (choi_pred - choi_true))))

            prob = cp.Problem(obj, constraints)
            prob.solve()

            return float(2 * prob.value)
        except ImportError:
            pass

    # Approximate using trace distance (lower bound)
    diff = choi_pred - choi_true
    eigenvalues = jnp.linalg.eigvalsh(diff @ diff.conj().T)
    return float(jnp.sqrt(jnp.sum(eigenvalues)))


def average_gate_fidelity(choi_true: jnp.ndarray, choi_pred: jnp.ndarray) -> float:
    """Compute average gate fidelity.

    Args:
        choi_true: True Choi matrix
        choi_pred: Predicted Choi matrix

    Returns:
        Average gate fidelity
    """
    d = int(np.sqrt(choi_true.shape[0]))

    # Process fidelity
    F_proc = process_fidelity(choi_true, choi_pred)

    # Average gate fidelity formula
    F_avg = (d * F_proc + 1) / (d + 1)

    return float(F_avg)


def reconstruction_error(data_true: jnp.ndarray, data_pred: jnp.ndarray) -> Dict[str, float]:
    """Compute various reconstruction error metrics.

    Args:
        data_true: True measurement data
        data_pred: Predicted measurement data

    Returns:
        Dictionary of error metrics
    """
    # Mean squared error
    mse = float(jnp.mean((data_true - data_pred) ** 2))

    # Mean absolute error
    mae = float(jnp.mean(jnp.abs(data_true - data_pred)))

    # Root mean squared error
    rmse = float(jnp.sqrt(mse))

    # Relative error
    rel_error = float(jnp.linalg.norm(data_true - data_pred) / (jnp.linalg.norm(data_true) + 1e-10))

    # R-squared score
    ss_res = jnp.sum((data_true - data_pred) ** 2)
    ss_tot = jnp.sum((data_true - jnp.mean(data_true)) ** 2)
    r2_score = float(1 - (ss_res / (ss_tot + 1e-10)))

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'relative_error': rel_error,
        'r2_score': r2_score
    }


def plot_attention_maps(attention_weights: List[jnp.ndarray],
                       layer_indices: Optional[List[int]] = None,
                       save_path: Optional[str] = None):
    """Visualize attention maps from transformer layers.

    Args:
        attention_weights: List of attention weights from each layer
        layer_indices: Which layers to plot (None = all)
        save_path: Path to save the figure
    """
    if layer_indices is None:
        layer_indices = range(len(attention_weights))

    n_layers = len(layer_indices)

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layer_indices):
        attn = attention_weights[layer_idx]

        # Average over batch and heads
        if attn.ndim == 4:  # (batch, heads, seq, seq)
            attn_avg = jnp.mean(attn, axis=(0, 1))
        elif attn.ndim == 3:  # (heads, seq, seq)
            attn_avg = jnp.mean(attn, axis=0)
        else:
            attn_avg = attn

        # Plot heatmap
        im = axes[idx].imshow(attn_avg, cmap='hot', aspect='auto')
        axes[idx].set_title(f'Layer {layer_idx + 1}')
        axes[idx].set_xlabel('Key Position')
        axes[idx].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[idx])

    plt.suptitle('Transformer Attention Maps')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_attention_entropy_evolution(attention_entropies: List[float],
                                    save_path: Optional[str] = None):
    """Plot evolution of attention entropy during training.

    Args:
        attention_entropies: List of entropy values over training steps
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    steps = np.arange(len(attention_entropies))
    ax.plot(steps, attention_entropies, 'b-', linewidth=2, label='Attention Entropy')

    # Add smoothed trend
    if len(attention_entropies) > 10:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(attention_entropies, sigma=5)
        ax.plot(steps, smoothed, 'r--', linewidth=2, alpha=0.7, label='Smoothed Trend')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Entropy')
    ax.set_title('Attention Entropy Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_measurement_selection(selected_indices: List[int],
                                   n_probes: int, n_measurements: int,
                                   save_path: Optional[str] = None):
    """Visualize which measurements were selected by adaptive sampling.

    Args:
        selected_indices: List of selected measurement indices
        n_probes: Number of probe states
        n_measurements: Number of measurement operators
        save_path: Path to save the figure
    """
    # Convert linear indices to 2D grid
    selection_matrix = np.zeros((n_probes, n_measurements))

    for idx in selected_indices:
        probe_idx = idx // n_measurements
        meas_idx = idx % n_measurements
        selection_matrix[probe_idx, meas_idx] += 1

    # Normalize to get selection frequency
    selection_freq = selection_matrix / len(selected_indices)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(selection_freq, cmap='YlOrRd', cbar_kws={'label': 'Selection Frequency'},
                xticklabels=range(n_measurements), yticklabels=range(n_probes))
    plt.xlabel('Measurement Index')
    plt.ylabel('Probe Index')
    plt.title('Adaptive Measurement Selection Pattern')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_convergence_comparison(losses_standard: List[float],
                               losses_transformer: List[float],
                               labels: Tuple[str, str] = ('Standard GD', 'Transformer GD'),
                               save_path: Optional[str] = None):
    """Compare convergence of standard vs transformer-enhanced gradient descent.

    Args:
        losses_standard: Loss values for standard method
        losses_transformer: Loss values for transformer method
        labels: Labels for the two methods
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(max(len(losses_standard), len(losses_transformer)))

    if losses_standard:
        ax.semilogy(steps[:len(losses_standard)], losses_standard, 'b-',
                   linewidth=2, label=labels[0])

    if losses_transformer:
        ax.semilogy(steps[:len(losses_transformer)], losses_transformer, 'r-',
                   linewidth=2, label=labels[1])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add improvement annotation
    if losses_standard and losses_transformer:
        final_improvement = (losses_standard[-1] - losses_transformer[-1]) / losses_standard[-1] * 100
        ax.text(0.6, 0.95, f'Improvement: {final_improvement:.1f}%',
               transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_uncertainty_evolution(uncertainties: Dict[str, List[float]],
                              save_path: Optional[str] = None):
    """Plot evolution of uncertainty estimates during training.

    Args:
        uncertainties: Dictionary mapping Kraus operator indices to uncertainty lists
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for kraus_idx, uncertainty_list in uncertainties.items():
        steps = np.arange(len(uncertainty_list))
        ax.plot(steps, uncertainty_list, linewidth=2, label=f'Kraus {kraus_idx}')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Uncertainty')
    ax.set_title('Uncertainty Evolution for Kraus Operators')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


class MetricsTracker:
    """Track and log metrics during training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_history = {
            'loss': [],
            'fidelity': [],
            'attention_entropy': [],
            'uncertainties': {},
            'selected_measurements': []
        }

    def update(self, **kwargs):
        """Update metrics with new values.

        Args:
            **kwargs: Metric names and values to update
        """
        for key, value in kwargs.items():
            if key == 'uncertainties':
                # Special handling for uncertainty dict
                for k, v in value.items():
                    if k not in self.metrics_history['uncertainties']:
                        self.metrics_history['uncertainties'][k] = []
                    self.metrics_history['uncertainties'][k].append(v)
            elif key in self.metrics_history:
                self.metrics_history[key].append(value)

    def get_history(self, metric: str) -> List:
        """Get history for a specific metric.

        Args:
            metric: Name of the metric

        Returns:
            List of metric values
        """
        return self.metrics_history.get(metric, [])

    def save(self, filepath: str):
        """Save metrics history to file.

        Args:
            filepath: Path to save the metrics
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics_history, f)

    def load(self, filepath: str):
        """Load metrics history from file.

        Args:
            filepath: Path to load the metrics from
        """
        import pickle
        with open(filepath, 'rb') as f:
            self.metrics_history = pickle.load(f)