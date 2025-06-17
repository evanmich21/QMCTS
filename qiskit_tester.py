#!/usr/bin/env python3
import sys
print("Using Python executable:", sys.executable)

# Check if Qiskit is installed
try:
    from qiskit import QuantumCircuit, execute, Aer, transpile
    from qiskit.visualization import plot_histogram
except ImportError:
    print("Qiskit is not installed. Please install it using 'pip install qiskit'")
    sys.exit(1)

def main():
    # Create a quantum circuit with 1 qubit and 1 classical bit.
    circuit = QuantumCircuit(1, 1)
    
    # Apply a Hadamard gate to put the qubit in a superposition.
    circuit.h(0)
    
    # Measure the qubit.
    circuit.measure(0, 0)
    
    # Use Aer's qasm_simulator.
    simulator = Aer.get_backend('qasm_simulator')
    
    # Transpile the circuit for the simulator.
    compiled_circuit = transpile(circuit, simulator)
    
    # Execute the circuit on the simulator with 1024 shots.
    job = execute(compiled_circuit, simulator, shots=1024)
    
    # Retrieve the results.
    result = job.result()
    counts = result.get_counts(circuit)
    print("Measurement results:", counts)
    
    # Optionally, create and save a histogram if matplotlib is available.
    try:
        import matplotlib.pyplot as plt
        plot_histogram(counts).savefig("qiskit_histogram.png")
        print("Histogram saved as qiskit_histogram.png")
    except ImportError:
        print("matplotlib is not installed; skipping histogram plot.")

if __name__ == '__main__':
    main()
