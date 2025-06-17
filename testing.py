import pennylane as qml
from pennylane import numpy as np

# Create a PennyLane device using Qiskit's Aer simulator.
# (If you run into issues with "qiskit.aer", try "qiskit.basicaer".)
dev = qml.device("qiskit.aer", wires=2)

@qml.qnode(dev)
def simple_quantum_circuit(theta):
    """
    A simple parameterized quantum circuit:
      - Applies a Hadamard on qubit 0.
      - Applies an RX rotation by angle theta on qubit 0.
      - Applies a CNOT gate (control qubit 0, target qubit 1).
    Returns expectation values of PauliZ on both qubits.
    """
    qml.Hadamard(wires=0)
    qml.RX(theta, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

# Test the circuit with a parameter value.
theta_val = np.pi / 4
exp_vals = simple_quantum_circuit(theta_val)

print("Expectation values for qubit 0 and qubit 1:")
print(exp_vals)
