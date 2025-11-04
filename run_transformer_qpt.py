#!/usr/bin/env python
"""
Quick start script to test Transformer-enhanced QPT implementation.
Run this to verify everything is working correctly.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

from qutip import *
import argparse
from tqdm.auto import tqdm

# Import our modules
from gd_qpt.core import tensor_product_list, convert_to_jax, choi
from gd_qpt.transformer_gd import TransformerGradientDescent
from gd_qpt.transformer_metrics import process_fidelity, average_gate_fidelity
from gd_qpt.gd import GradientDescent


def create_test_process(n_qubits=2, process_type='cnot', noise_level=0.1):
    """Create a test quantum process."""
    N = 2**n_qubits

    if process_type == 'cnot' and n_qubits == 2:
        # CNOT gate
        cnot = tensor(sigmax(), sigmax()) + tensor(sigmaz(), qeye(2))
        U = (cnot / 2).expm()
    elif process_type == 'hadamard':
        # Hadamard on all qubits
        H_single = hadamard_transform(1)
        U = tensor(*[H_single for _ in range(n_qubits)])
    elif process_type == 'random':
        # Random unitary
        U = rand_unitary(N, density=0.75)
    else:
        # Identity
        U = qeye(N)

    # Create noisy Kraus operators
    kraus_ops = []
    kraus_ops.append(np.sqrt(1 - noise_level) * U.full())

    # Add depolarizing noise
    if noise_level > 0:
        # Simple depolarizing channel
        for _ in range(3):
            kraus_ops.append(np.sqrt(noise_level/3) * rand_unitary(N, density=0.5).full())

    return kraus_ops


def generate_pauli_measurements(n_qubits):
    """Generate Pauli measurement basis."""
    # Single qubit Pauli eigenstates
    pauli_states_1q = [
        basis(2, 0),  # |0>
        basis(2, 1),  # |1>
        (basis(2, 0) + basis(2, 1)).unit(),  # |+>
        (basis(2, 0) - basis(2, 1)).unit(),  # |->
        (basis(2, 0) + 1j*basis(2, 1)).unit(),  # |+i>
        (basis(2, 0) - 1j*basis(2, 1)).unit(),  # |-i>
    ]

    # Create n-qubit states
    probes = tensor_product_list(pauli_states_1q, n_qubits)
    measurements = tensor_product_list(pauli_states_1q, n_qubits)

    return convert_to_jax(probes), convert_to_jax(measurements)


def generate_measurement_data(kraus_ops, probes, measurements, noise_std=0.001):
    """Generate synthetic measurement data."""
    n_probes = probes.shape[0]
    n_measurements = measurements.shape[0]
    data = np.zeros((n_probes, n_measurements))

    for i, probe in enumerate(probes):
        rho_in = np.outer(probe, probe.conj())
        rho_out = sum([K @ rho_in @ K.conj().T for K in kraus_ops])

        for j, meas in enumerate(measurements):
            proj = np.outer(meas, meas.conj())
            prob = np.real(np.trace(proj @ rho_out))
            data[i, j] = prob

    # Add noise
    data += np.random.normal(0, noise_std, data.shape)
    data = np.clip(data, 0, 1)

    return jnp.array(data)


def main(args):
    """Main execution function."""
    print("="*60)
    print("Transformer-Enhanced Quantum Process Tomography")
    print("="*60)

    # Setup
    n_qubits = args.n_qubits
    N = 2**n_qubits
    print(f"\nSystem: {n_qubits} qubits (Hilbert space dimension: {N})")

    # Create test process
    print(f"Creating test process: {args.process_type} with {args.noise_level*100:.1f}% noise")
    kraus_true = create_test_process(n_qubits, args.process_type, args.noise_level)
    num_kraus = len(kraus_true)
    choi_true = choi(np.array(kraus_true))

    # Generate measurements
    print("Generating Pauli measurements...")
    probes, measurements = generate_pauli_measurements(n_qubits)
    print(f"  Probes: {probes.shape[0]}, Measurements: {measurements.shape[0]}")

    # Generate data
    print("Generating measurement data...")
    data = generate_measurement_data(kraus_true, probes, measurements)

    # Use only a fraction of measurements if specified
    if args.measurement_fraction < 1.0:
        n_total = data.shape[0] * data.shape[1]
        n_use = int(args.measurement_fraction * n_total)
        print(f"Using {args.measurement_fraction*100:.0f}% of measurements ({n_use}/{n_total})")

        # Create sparse data by masking
        mask = np.zeros(n_total, dtype=bool)
        mask[np.random.choice(n_total, n_use, replace=False)] = True
        mask = mask.reshape(data.shape)
        data = jnp.where(mask, data, 0)

    # Configure transformer
    transformer_config = {
        'd_model': min(256, 64 * n_qubits),  # Scale with system size
        'n_heads': min(8, 2**n_qubits),
        'n_layers': min(6, n_qubits + 2),
        'd_ff': min(1024, 256 * n_qubits),
        'dropout_rate': 0.1
    }

    print(f"\nTransformer configuration:")
    for key, value in transformer_config.items():
        print(f"  {key}: {value}")

    # Train transformer model
    if not args.skip_transformer:
        print("\n" + "-"*40)
        print("Training Transformer-Enhanced Model...")
        print("-"*40)

        model_transformer = TransformerGradientDescent(
            N=N,
            num_kraus=num_kraus,
            lr=args.learning_rate,
            alpha=0.995,
            transformer_config=transformer_config
        )

        choi_transformer = model_transformer.fit(
            data,
            probes,
            measurements,
            batch_size=min(32, data.shape[0] // 2),
            maxiters=args.max_iters,
            use_transformer=True,
            pretrain_epochs=args.pretrain_epochs
        )

        # Evaluate
        fidelity_transformer = process_fidelity(choi_true, choi_transformer)
        avg_fidelity_transformer = average_gate_fidelity(choi_true, choi_transformer)

        print(f"\nTransformer Results:")
        print(f"  Process Fidelity: {fidelity_transformer:.6f}")
        print(f"  Average Gate Fidelity: {avg_fidelity_transformer:.6f}")

    # Train standard model for comparison
    if not args.skip_standard:
        print("\n" + "-"*40)
        print("Training Standard GD Model (for comparison)...")
        print("-"*40)

        model_standard = GradientDescent(
            N=N,
            num_kraus=num_kraus,
            lr=args.learning_rate,
            alpha=0.995
        )

        choi_standard = model_standard.fit(
            data,
            probes,
            measurements,
            batch_size=min(32, data.shape[0] // 2),
            maxiters=args.max_iters
        )

        # Evaluate
        fidelity_standard = process_fidelity(choi_true, choi_standard)
        avg_fidelity_standard = average_gate_fidelity(choi_true, choi_standard)

        print(f"\nStandard GD Results:")
        print(f"  Process Fidelity: {fidelity_standard:.6f}")
        print(f"  Average Gate Fidelity: {avg_fidelity_standard:.6f}")

    # Comparison
    if not args.skip_transformer and not args.skip_standard:
        print("\n" + "="*40)
        print("COMPARISON")
        print("="*40)
        print(f"Process Fidelity:")
        print(f"  Transformer: {fidelity_transformer:.6f}")
        print(f"  Standard:    {fidelity_standard:.6f}")
        print(f"  Improvement: {(fidelity_transformer - fidelity_standard)*100:.2f}%")

        print(f"\nAverage Gate Fidelity:")
        print(f"  Transformer: {avg_fidelity_transformer:.6f}")
        print(f"  Standard:    {avg_fidelity_standard:.6f}")
        print(f"  Improvement: {(avg_fidelity_transformer - avg_fidelity_standard)*100:.2f}%")

    print("\n" + "="*60)
    print("✅ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Transformer-Enhanced QPT')
    parser.add_argument('--n_qubits', type=int, default=2,
                        help='Number of qubits (default: 2)')
    parser.add_argument('--process_type', type=str, default='cnot',
                        choices=['cnot', 'hadamard', 'random', 'identity'],
                        help='Type of quantum process to test')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level (0-1, default: 0.1)')
    parser.add_argument('--measurement_fraction', type=float, default=1.0,
                        help='Fraction of measurements to use (0-1, default: 1.0)')
    parser.add_argument('--max_iters', type=int, default=300,
                        help='Maximum iterations for gradient descent')
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='Number of pre-training epochs for transformer')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Initial learning rate')
    parser.add_argument('--skip_transformer', action='store_true',
                        help='Skip transformer model training')
    parser.add_argument('--skip_standard', action='store_true',
                        help='Skip standard model training')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease make sure all dependencies are installed:")
        print("  pip install -r requirements_transformer.txt")
        raise