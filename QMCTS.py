import math
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, BasicAer

# --- Quantum Randomness Helper ---

from qiskit import QuantumCircuit, BasicAer

def quantum_choice(options):
    """
    Uses BasicAer's qasm_simulator to randomly select an element from the list 'options'.
    """
    from math import ceil, log2
    n = len(options)
    num_qubits = ceil(log2(n))
    backend = BasicAer.get_backend('qasm_simulator')
    
    while True:
        qc = QuantumCircuit(num_qubits, num_qubits)
        # Put each qubit into superposition
        for q in range(num_qubits):
            qc.h(q)
        qc.measure(range(num_qubits), range(num_qubits))
        
        # Run the circuit on BasicAer's simulator
        job = backend.run(qc, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        # Get the first (and only) measured bitstring
        measured_str = list(counts.keys())[0]
        index = int(measured_str, 2)
        if index < n:
            return options[index]




# --- Maze Definition ---

class QuantumMaze:
    def __init__(self):
        self.size = 4
        # 0: free cell, 1: wall.
        self.layout = [
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0]
        ]
        self.start = (0, 0)
        self.goal = (3, 3)
    
    def valid_move(self, pos):
        x, y = pos
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.layout[y][x] == 0

    def available_moves(self, pos):
        x, y = pos
        moves = {}
        directions = {
            'U': (x, y - 1),
            'D': (x, y + 1),
            'L': (x - 1, y),
            'R': (x + 1, y)
        }
        for d, new_pos in directions.items():
            if self.valid_move(new_pos):
                moves[d] = new_pos
        # Return as a list of (direction, new_pos) tuples.
        return list(moves.items())

    def reached_goal(self, pos):
        return pos == self.goal

# --- Tree Node for QMCTS ---

class QuantumNode:
    def __init__(self, pos, parent=None, move=None):
        self.pos = pos              # Current cell (x,y)
        self.parent = parent        # Parent node
        self.move = move            # Move taken from the parent to reach here
        self.children = []          # List of child nodes
        self.visits = 0             # Visit count
        self.wins = 0               # Win count (number of simulations that reached the goal)
        self.unexplored = None      # Unexpanded moves from this node

    def path(self):
        """Return the path from the root to this node."""
        p = []
        curr = self
        while curr:
            p.append(curr.pos)
            curr = curr.parent
        return p[::-1]
    
    def is_fully_expanded(self, maze):
        if self.unexplored is None:
            self.unexplored = maze.available_moves(self.pos).copy()
        return len(self.unexplored) == 0

    def expand(self, maze):
        if self.unexplored is None:
            self.unexplored = maze.available_moves(self.pos).copy()
        if self.unexplored:
            # Use quantum_choice to select which move to expand.
            idx = quantum_choice(list(range(len(self.unexplored))))
            move, new_pos = self.unexplored.pop(idx)
            child = QuantumNode(new_pos, parent=self, move=move)
            self.children.append(child)
            return child
        return None

    def best_child(self, c_param=1.4):
        best_score = -float("inf")
        best_child = None
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                score = (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

# --- Quantum Rollout (Simulation) ---

def quantum_rollout(start_pos, maze, max_steps=50):
    current = start_pos
    for _ in range(max_steps):
        if maze.reached_goal(current):
            return 1  # Successful simulation
        moves = maze.available_moves(current)
        if not moves:
            break
        # Use quantum randomness to choose the next move.
        move, next_pos = quantum_choice(moves)
        current = next_pos
    return 0  # Failed to reach the goal within max_steps

# --- Quantum Monte Carlo Tree Search ---

def quantum_mcts(maze, iterations):
    root = QuantumNode(maze.start)
    solved_iteration = None

    for it in range(iterations):
        node = root

        # --- Selection ---
        while (not maze.reached_goal(node.pos)) and node.is_fully_expanded(maze) and node.children:
            node = node.best_child()

        # --- Expansion ---
        if not maze.reached_goal(node.pos):
            if not node.is_fully_expanded(maze):
                node = node.expand(maze)

        # Record the iteration when the goal is first reached.
        if maze.reached_goal(node.pos) and solved_iteration is None:
            solved_iteration = it + 1
            print(f"*** Quantum goal reached at iteration {solved_iteration} ***")

        # --- Simulation (Rollout) ---
        result = quantum_rollout(node.pos, maze)
        
        # --- Backpropagation ---
        temp = node
        while temp is not None:
            temp.visits += 1
            if result == 1:
                temp.wins += 1
            temp = temp.parent

    return root, solved_iteration

def find_quantum_solution(node, maze):
    """Recursively searches for a node that reached the goal and returns its path."""
    if maze.reached_goal(node.pos):
        return node.path()
    for child in node.children:
        sol = find_quantum_solution(child, maze)
        if sol is not None:
            return sol
    return None

# --- Visualization (same colors as before) ---

def plot_quantum_maze(maze, solution_path=None):
    grid = np.array(maze.layout)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid, cmap="binary", origin="upper")

    sx, sy = maze.start
    gx, gy = maze.goal
    ax.scatter(sx, sy, c="green", s=200, marker="o", label="Start")
    ax.scatter(gx, gy, c="red", s=200, marker="x", label="Goal")

    if solution_path:
        xs = [p[0] for p in solution_path]
        ys = [p[1] for p in solution_path]
        ax.plot(xs, ys, c="blue", linewidth=2, label="Path")
        ax.scatter(xs, ys, c="blue", s=50)

    ax.set_xticks(np.arange(-0.5, maze.size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, maze.size - 0.5)
    ax.set_ylim(maze.size - 0.5, -0.5)
    ax.legend(loc="upper right")
    plt.title("Quantum Maze and Solution")
    plt.show()

# --- Main Function ---

def main():
    qmaze = QuantumMaze()
    iterations = 1000
    print("Starting Quantum Monte Carlo Tree Search (QMCTS)...\n")
    root, solved_iter = quantum_mcts(qmaze, iterations)
    solution = find_quantum_solution(root, qmaze)
    
    if solution:
        print("\nQuantum solution path found:")
        for pos in solution:
            print(pos)
        if solved_iter:
            print(f"\nThe target was first reached on iteration {solved_iter}.")
    else:
        print("\nNo solution was discovered.")
    
    plot_quantum_maze(qmaze, solution)

if __name__ == "__main__":
    main()
