import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml

# ==========================
# Unified Maze Definition (5x5)
# ==========================
class Maze5x5:
    def __init__(self):
        self.rows = 5
        self.cols = 5
        # 0 = free cell, 1 = wall.
        self.grid = [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        self.start = (0, 0)
        self.goal = (4, 4)
    
    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        return self.grid[y][x] == 0
    
    def get_moves(self, pos):
        x, y = pos
        moves = []
        directions = {
            "up": (x, y - 1),
            "down": (x, y + 1),
            "left": (x - 1, y),
            "right": (x + 1, y)
        }
        for d, new_pos in directions.items():
            if self.is_valid(new_pos):
                moves.append((d, new_pos))
        return moves
    
    def reached_target(self, pos):
        return pos == self.goal

# ==========================
# Randomness Selector Functions
# ==========================
def classical_choice(options):
    return random.choice(options)

def quantum_random_choice(options):
    """
    Uses a precompiled quantum circuit (via PennyLane) to generate quantum randomness.
    """
    n = len(options)
    if n == 1:
        return options[0]
    num_qubits = ceil(log2(n))
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        for i in range(num_qubits):
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(num_qubits))
    
    probs = circuit()
    indices = list(range(2**num_qubits))
    chosen = np.random.choice(indices, p=probs)
    if chosen < n:
        return options[chosen]
    else:
        return quantum_random_choice(options)

# ==========================
# Unified MCTS Node Definition
# ==========================
class MCTSNode:
    def __init__(self, pos, parent=None, move_taken=None):
        self.position = pos            # Maze cell (x, y)
        self.parent = parent           # Parent node
        self.move_taken = move_taken   # Move taken to reach this node
        self.children = []
        self.visit_count = 0
        self.win_count = 0
        self.available_actions = None  # Moves not yet tried
    
    def build_path(self):
        path = []
        curr = self
        while curr:
            path.append(curr.position)
            curr = curr.parent
        return path[::-1]
    
    def is_fully_expanded(self, maze):
        if self.available_actions is None:
            self.available_actions = maze.get_moves(self.position).copy()
        return len(self.available_actions) == 0
    
    def add_child(self, maze, selector):
        if self.available_actions is None:
            self.available_actions = maze.get_moves(self.position).copy()
        if self.available_actions:
            chosen_move = selector(self.available_actions)
            self.available_actions.remove(chosen_move)
            child = MCTSNode(chosen_move[1], parent=self, move_taken=chosen_move[0])
            self.children.append(child)
            return child
        return None
    
    def choose_child(self, exploration_factor=1.4):
        best_value = -float("inf")
        chosen_child = None
        for child in self.children:
            if child.visit_count == 0:
                score = float("inf")
            else:
                score = (child.win_count / child.visit_count) + \
                        exploration_factor * math.sqrt(math.log(self.visit_count) / child.visit_count)
            if score > best_value:
                best_value = score
                chosen_child = child
        return chosen_child

# ==========================
# Unified Simulation Function
# ==========================
def simulate_trial(starting_pos, maze, selector, max_steps=50):
    current = starting_pos
    for _ in range(max_steps):
        if maze.reached_target(current):
            return 1
        possible_moves = maze.get_moves(current)
        if not possible_moves:
            break
        action, next_pos = selector(possible_moves)
        current = next_pos
    return 0

# ==========================
# Unified MCTS Search Function
# ==========================
def mcts_search(maze, total_iterations, selector):
    root = MCTSNode(maze.start)
    solution_iteration = None
    
    for iteration in range(total_iterations):
        current_node = root
        
        while (not maze.reached_target(current_node.position)) and \
              current_node.is_fully_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()
        
        if not maze.reached_target(current_node.position):
            if not current_node.is_fully_expanded(maze):
                current_node = current_node.add_child(maze, selector)
        
        if maze.reached_target(current_node.position) and solution_iteration is None:
            solution_iteration = iteration + 1
            print(f"*** Goal reached at iteration {solution_iteration} ***")
        
        trial_outcome = simulate_trial(current_node.position, maze, selector)
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent
    
    return root, solution_iteration

def extract_solution(node, maze):
    if maze.reached_target(node.position):
        return node.build_path()
    for child in node.children:
        sol = extract_solution(child, maze)
        if sol is not None:
            return sol
    return None

# ==========================
# DRIVER FUNCTIONS FOR COMPARISON
# ==========================
def run_classical_mcts(iterations):
    maze = Maze5x5()
    start_time = time.time()
    root, sol_iter = mcts_search(maze, iterations, classical_choice)
    end_time = time.time()
    solution = extract_solution(root, maze)
    return {
        "algorithm": "Classical MCTS",
        "solution": solution,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

def run_quantum_mcts(iterations):
    maze = Maze5x5()
    start_time = time.time()
    root, sol_iter = mcts_search(maze, iterations, quantum_random_choice)
    end_time = time.time()
    solution = extract_solution(root, maze)
    return {
        "algorithm": "Quantum MCTS",
        "solution": solution,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

# ==========================
# MAIN COMPARISON DRIVER
# ==========================
def main():
    iterations = 200  # Adjust as desired.
    
    print("Running Classical MCTS...")
    classical_result = run_classical_mcts(iterations)
    print("\nRunning Quantum MCTS...")
    quantum_result = run_quantum_mcts(iterations)
    
    print("\n=== Comparison Summary ===")
    print("Classical MCTS:")
    print(f"  Iterations: {classical_result['iterations']}")
    print(f"  Running Time: {classical_result['time']:.2f} seconds")
    print(f"  Solution: {classical_result['solution']}")
    print(f"  First reached at iteration: {classical_result['solution_iteration']}")
    
    print("\nQuantum MCTS:")
    print(f"  Iterations: {quantum_result['iterations']}")
    print(f"  Running Time: {quantum_result['time']:.2f} seconds")
    print(f"  Solution: {quantum_result['solution']}")
    print(f"  First reached at iteration: {quantum_result['solution_iteration']}")
    
if __name__ == "__main__":
    main()
