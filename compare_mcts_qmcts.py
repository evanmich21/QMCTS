import math
import random
import numpy as np
import matplotlib.pyplot as plt
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
# Quantum Randomness Helper (using PennyLane)
# ==========================
def quantum_random_choice(options):
    """
    Uses a quantum circuit (with PennyLane's default.qubit simulator)
    to randomly select one element from 'options'. Builds a circuit with
    enough wires to represent indices 0,...,n-1, and returns the chosen option.
    Uses rejection sampling if the measured index is out of range.
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

# ------------------------------
# Helper: Manhattan Distance
# ------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

# ==========================
# CLASSICAL MCTS IMPLEMENTATION
# ==========================

class TreeNode:
    def __init__(self, pos, parent=None, move_taken=None):
        self.position = pos              # Maze cell (x, y)
        self.parent = parent             # Parent node in the tree
        self.move_taken = move_taken     # Move taken from parent to reach this node
        self.children = []               # List of child nodes
        self.visit_count = 0             # Visit count
        self.win_count = 0               # Cumulative reward
        self.available_actions = None    # Moves yet to try from this node

    def build_path(self):
        path = []
        curr = self
        while curr:
            path.append(curr.position)
            curr = curr.parent
        return path[::-1]

    def is_completely_expanded(self, maze):
        if self.available_actions is None:
            self.available_actions = maze.get_moves(self.position).copy()
        return len(self.available_actions) == 0

    def add_child(self, maze):
        if self.available_actions is None:
            self.available_actions = maze.get_moves(self.position).copy()
        if self.available_actions:
            idx = random.randrange(len(self.available_actions))
            action, new_pos = self.available_actions.pop(idx)
            child = TreeNode(new_pos, parent=self, move_taken=action)
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

def simulate_trial(starting_pos, maze, max_steps=50):
    current = starting_pos
    for _ in range(max_steps):
        if maze.reached_target(current):
            return 1
        possible_moves = maze.get_moves(current)
        if not possible_moves:
            break
        action, next_pos = random.choice(possible_moves)
        current = next_pos
    return 0

def mct_search(maze, total_iterations):
    root_node = TreeNode(maze.start)
    solution_iteration = None

    for iteration in range(total_iterations):
        current_node = root_node

        # Selection
        while (not maze.reached_target(current_node.position)) and current_node.is_completely_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()

        # Expansion
        if not maze.reached_target(current_node.position):
            if not current_node.is_completely_expanded(maze):
                current_node = current_node.add_child(maze)

        # Record first solution
        if maze.reached_target(current_node.position) and solution_iteration is None:
            solution_iteration = iteration + 1
            print(f"*** Classical goal reached at iteration {solution_iteration} ***")

        # Simulation
        trial_outcome = simulate_trial(current_node.position, maze)
        # Backpropagation
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent

    return root_node, solution_iteration

def extract_classical_solution(node, maze):
    if maze.reached_target(node.position):
        return node.build_path()
    for child in node.children:
        sol = extract_classical_solution(child, maze)
        if sol is not None:
            return sol
    return None

# ==========================
# QUANTUM MCTS (QMCTS) IMPLEMENTATION
# (Enhanced with step penalty, Manhattan filtering, and no immediate backtracking)
# ==========================

class QuantumNode:
    def __init__(self, pos, parent=None, move=None):
        self.state = pos              # Maze cell (x, y)
        self.parent = parent
        self.move = move              # Move taken to reach this state
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = None

    def is_fully_expanded(self, maze):
        if self.untried_moves is None:
            self.untried_moves = maze.get_moves(self.state).copy()
        return len(self.untried_moves) == 0

    def expand(self, maze):
        if self.untried_moves is None:
            self.untried_moves = maze.get_moves(self.state).copy()
        if self.untried_moves:
            chosen_move = quantum_random_choice(self.untried_moves)
            self.untried_moves.remove(chosen_move)
            child = QuantumNode(chosen_move[1], parent=self, move=chosen_move[0])
            self.children.append(child)
            return child
        return None

    def best_child(self, c_param=1.4):
        best = None
        best_score = -float("inf")
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                score = (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    def path(self):
        p = []
        curr = self
        while curr:
            p.append(curr.state)
            curr = curr.parent
        return p[::-1]

def simulate_qmcts(maze, state, max_steps=100):
    current = state
    steps = 0
    visited = {}
    path = [current]
    while current != maze.goal and steps < max_steps:
        moves = maze.get_moves(current)
        # Filter: if moves reducing Manhattan distance exist, choose them.
        current_dist = manhattan_distance(current, maze.goal)
        reducing_moves = [m for m in moves if manhattan_distance(m[1], maze.goal) < current_dist]
        if reducing_moves:
            moves = reducing_moves
        # Prevent immediate backtracking.
        if len(path) >= 2:
            prev_state = path[-2]
            non_back_moves = [m for m in moves if m[1] != prev_state]
            if non_back_moves:
                moves = non_back_moves
        if not moves:
            break
        chosen_move = quantum_random_choice(moves)
        current = chosen_move[1]
        path.append(current)
        steps += 1
        visited[current] = visited.get(current, 0) + 1
    max_dist = maze.rows + maze.cols - 2  # For 5x5, max Manhattan = 8.
    dist = manhattan_distance(current, maze.goal)
    if current == maze.goal:
        reward = 10
    else:
        base_reward = (max_dist - dist) / max_dist - 0.5
        penalty = sum((count - 1) * 0.2 for count in visited.values() if count > 1)
        reward = base_reward - penalty
    step_penalty = 0.1 * steps
    final_reward = reward - step_penalty
    return final_reward

def mcts_qmcts(maze, iterations):
    root = QuantumNode(maze.start)
    solved_iteration = None
    best_solution_steps = None
    best_solution_iteration = None
    for i in range(iterations):
        node = root
        # Selection
        while node.state != maze.goal and node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
        # Expansion
        if node.state != maze.goal:
            node = node.expand(maze)
        # Check if goal reached.
        if node.state == maze.goal:
            path = extract_solution_qmcts(node, maze)
            steps_taken = len(path)
            if best_solution_steps is None or steps_taken < best_solution_steps:
                best_solution_steps = steps_taken
                best_solution_iteration = i + 1
                print(f"*** New best QMCTS solution at iteration {best_solution_iteration} with {steps_taken} steps ***")
            if solved_iteration is None:
                solved_iteration = i + 1
                print(f"*** QMCTS goal reached at iteration {solved_iteration} ***")
        result = simulate_qmcts(maze, node.state)
        temp = node
        while temp is not None:
            temp.visits += 1
            if result == 1:
                temp.wins += 1
            temp = temp.parent
    return root, solved_iteration, best_solution_iteration, best_solution_steps

def extract_solution_qmcts(node, maze):
    if node.state == maze.goal:
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]
    for child in node.children:
        sol = extract_solution_qmcts(child, maze)
        if sol is not None:
            return sol
    return None

def find_quantum_solution(root, maze):
    if maze.reached_target(root.state):
        return root.path()
    for child in root.children:
        sol = find_quantum_solution(child, maze)
        if sol is not None:
            return sol
    return None

# ==========================
# COMPARISON DRIVER FUNCTIONS
# ==========================

def run_classical_mcts(iterations):
    maze = Maze5x5()  # Unified maze.
    start_time = time.time()
    root, sol_iter = mct_search(maze, iterations)
    end_time = time.time()
    solution = extract_classical_solution(root, maze)
    return {
        "algorithm": "Classical MCTS",
        "solution": solution,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

def run_quantum_mcts(iterations):
    maze = Maze5x5()  # Unified maze.
    start_time = time.time()
    root, sol_iter, best_iter, best_steps = mcts_qmcts(maze, iterations)
    end_time = time.time()
    solution = find_quantum_solution(root, maze)
    return {
        "algorithm": "Quantum MCTS",
        "solution": solution,
        "solution_iteration": sol_iter,
        "best_solution_iteration": best_iter,
        "best_solution_steps": best_steps,
        "iterations": iterations,
        "time": end_time - start_time
    }

def extract_classical_solution(node, maze):
    if maze.reached_target(node.position):
        return node.build_path()
    for child in node.children:
        sol = extract_classical_solution(child, maze)
        if sol is not None:
            return sol
    return None

# ==========================
# MAIN COMPARISON DRIVER
# ==========================
def main():
    iterations = 1000  # Adjust iterations as desired.
    
    print("Running Classical MCTS...")
    classical_result = run_classical_mcts(iterations)
    print("\nRunning Quantum MCTS...")
    quantum_result = run_quantum_mcts(iterations)
    
    print("\n=== Comparison Summary ===")
    print("Classical MCTS:")
    print(f"  Iterations: {classical_result['iterations']}")
    print(f"  Running Time: {classical_result['time']:.2f} seconds")
    print(f"  Solution: {classical_result['solution']}")
    print(f"  First found at iteration: {classical_result['solution_iteration']}")
    
    print("\nQuantum MCTS:")
    print(f"  Iterations: {quantum_result['iterations']}")
    print(f"  Running Time: {quantum_result['time']:.2f} seconds")
    print(f"  Solution: {quantum_result['solution']}")
    print(f"  First reached at iteration: {quantum_result['solution_iteration']}")
    print(f"  Best (shortest) solution found at iteration: {quantum_result['best_solution_iteration']} with {quantum_result['best_solution_steps']} steps")
    
if __name__ == "__main__":
    main()
