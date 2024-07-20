import numpy as np
from scipy.linalg import expm, logm

# Function to generate a random unitary matrix (Lie group element from U(n))
def random_unitary(n):
    z = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q @ np.diag(ph)

# Function to generate a random skew-Hermitian matrix (Lie algebra element)
def random_skew_hermitian(n):
    a = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    return a - a.conj().T

# Function to apply an error to a unitary matrix
def apply_error(unitary, error):
    return unitary @ expm(error)

# Function to detect and correct error
def correct_error(unitary_with_error, expected_unitary):
    error_estimate = logm(unitary_with_error @ np.linalg.inv(expected_unitary))
    corrected_unitary = unitary_with_error @ expm(-error_estimate)
    return corrected_unitary

# Parameters
n = 2  # Dimension of the unitary group U(n)
num_simulations = 5  # Number of simulations

# Simulation
for i in range(num_simulations):
    print(f"Simulation {i + 1}")
    
    # Generate a random initial unitary matrix
    U_expected = random_unitary(n)
    print("Expected Unitary (U_expected):")
    print(U_expected)
    
    # Generate a random skew-Hermitian error
    H_error = random_skew_hermitian(n)
    print("Applied Error (H_error):")
    print(H_error)
    
    # Apply the error to the unitary matrix
    U_actual = apply_error(U_expected, H_error)
    print("Unitary with Error (U_actual):")
    print(U_actual)
    
    # Correct the error
    U_corrected = correct_error(U_actual, U_expected)
    print("Corrected Unitary (U_corrected):")
    print(U_corrected)
    
    # Check the difference between the corrected unitary and the expected unitary
    difference = np.linalg.norm(U_corrected - U_expected)
    print(f"Difference after correction: {difference}")
    print("-" * 40)
