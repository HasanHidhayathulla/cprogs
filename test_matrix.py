import subprocess
import sys

def test_matrix_binary():
    print("Running matrix program execution test...")
    result = subprocess.run(["./matrix"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED: matrix returned exit code {result.returncode}")
        print("STDERR:", result.stderr)
        sys.exit(1)

    output = result.stdout

    expected_snippets = [
        "MATRIX OPERATIONS IN THE UNIVERSE DEMO",
        "A + B",
        "A - B",
        "2.5 * A",
        "A * B (Matrix Multiplication)",
        "A o B (Hadamard Product)",
        "B^T (Transpose of B)",
        "Trace(A): 11.0000",
        "Determinant(A): 27.0000",
        "Rank(A): 3",
        "Frobenius Norm(A): 7.1414",
        "Infinity Norm(A): 7.0000",
        "Is A Symmetric? Yes",
        "A^-1 (Inverse of A)",
        "A * A^-1 (Identity check)",
        "A^3 (Matrix Power 3)",
        "Kronecker Product (K1 (x) K2)",
        "LU Decomposition: L",
        "LU Decomposition: U",
        "QR Decomposition: Q",
        "QR Decomposition: R",
        "Eigenvalues (Diagonal)",
        "Eigenvectors (Columns)",
        "DEMO COMPLETED SUCCESSFULLY"
    ]

    missing = []
    for snippet in expected_snippets:
        if snippet not in output:
            missing.append(snippet)

    if missing:
        print(f"FAILED: Missing expected output snippets: {missing}")
        sys.exit(1)

    print("SUCCESS: All matrix operation outputs verified successfully.")

if __name__ == "__main__":
    test_matrix_binary()
