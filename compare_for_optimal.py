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
        # Grid layout: 0 = free cell, 1 = wall.
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
# Global Quantum Device and Precompiled QNode for QMCTS
# ==========================
GLOBAL_NUM_WIRES = 4
GLOBAL_DEV = qml.device("default.qubit", wires=GLOBAL_NUM_WIRES)

@qml.qnode(GLOBAL_DEV)
def precompiled_quantum_choice():
    for i in range(GLOBAL_NUM_WIRES):
        qml.Hadamard(wires=i)
    return qml.probs(wires=range(GLOBAL_NUM_WIRES))

# ==========================
# Randomness Selector Functions
# ==========================
def classical_choice(options):
    return random.choice(options)

def quantum_random_choice(options):
    n = len(options)
    if n == 1:
        return options[0]
    probs = precompiled_quantum_choice()[:2**GLOBAL_NUM_WIRES]
    indices = list(range(2**GLOBAL_NUM_WIRES))
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
        self.position = pos
        self.parent = parent
        self.move_taken = move_taken
        self.children = []
        self.visit_count = 0
        self.win_count = 0
        self.available_actions = None
    
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
# Unified MCTS Search Function (Run Until Optimal Path Found)
# ==========================
def mcts_search_optimal(maze, total_iterations, selector, optimal_length):
    root = MCTSNode(maze.start)
    solution_iteration = None
    best_path = None
    best_path_length = float('inf')
    
    for iteration in range(total_iterations):
        current_node = root
        
        while (not maze.reached_target(current_node.position)) and \
              current_node.is_fully_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()
        
        if not maze.reached_target(current_node.position):
            if not current_node.is_fully_expanded(maze):
                current_node = current_node.add_child(maze, selector)
        
        if maze.reached_target(current_node.position):
            path = current_node.build_path()
            path_length = len(path)
            if solution_iteration is None:
                solution_iteration = iteration + 1
                print(f"*** Goal reached at iteration {solution_iteration} ***")
            if path_length < best_path_length:
                best_path_length = path_length
                best_path = path
                print(f"*** New best solution (length {best_path_length}) at iteration {iteration + 1} ***")
            if best_path_length == optimal_length:
                print(f"*** Optimal solution found at iteration {iteration + 1} ***")
                return root, solution_iteration, best_path
        
        trial_outcome = simulate_trial(current_node.position, maze, selector)
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent
    
    return root, solution_iteration, best_path

# ==========================
# DRIVER FUNCTIONS FOR COMPARISON (Optimal)
# ==========================
def run_classical_mcts_optimal(iterations, optimal_length):
    maze = Maze5x5()
    start_time = time.time()
    root, sol_iter, best_path = mcts_search_optimal(maze, iterations, classical_choice, optimal_length)
    end_time = time.time()
    return {
        "algorithm": "Classical MCTS",
        "solution": best_path,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

def run_quantum_mcts_optimal(iterations, optimal_length):
    maze = Maze5x5()
    start_time = time.time()
    root, sol_iter, best_path = mcts_search_optimal(maze, iterations, quantum_random_choice, optimal_length)
    end_time = time.time()
    return {
        "algorithm": "Quantum MCTS",
        "solution": best_path,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

# ==========================
# MAIN COMPARISON DRIVER (Optimal)
# ==========================
def main_optimal():
    # For our Maze5x5, the optimal solution (shortest path) is known to have 8 nodes.
    # (For instance: (0,0) -> (1,0) -> (2,0) -> (3,0) -> (4,0) -> (4,1) -> (4,2) -> (4,3) -> (4,4))
    optimal_length = 8  
    iterations = 10000  # Increase iterations to allow finding the optimal path.
    
    print("Running Classical MCTS until optimal solution found...")
    classical_result = run_classical_mcts_optimal(iterations, optimal_length)
    print("\nRunning Quantum MCTS until optimal solution found...")
    quantum_result = run_quantum_mcts_optimal(iterations, optimal_length)
    
    print("\n=== Comparison Summary (Optimal) ===")
    print("Classical MCTS:")
    print(f"  Running Time: {classical_result['time']:.2f} seconds")
    print(f"  Best Solution: {classical_result['solution']}")
    print(f"  (Optimal solution length: {optimal_length} nodes)")
    
    print("\nQuantum MCTS:")
    print(f"  Running Time: {quantum_result['time']:.2f} seconds")
    print(f"  Best Solution: {quantum_result['solution']}")
    print(f"  (Optimal solution length: {optimal_length} nodes)")
    
if __name__ == "__main__":
    # Uncomment one of the following to run the desired test:
    
    # 1. Stop as soon as any solution is found:
    # def main_stop():
    #     iterations = 1000
    #     print("Running Classical MCTS (Stop on Goal)...")
    #     classical_result = run_classical_mcts(iterations)
    #     print("\nRunning Quantum MCTS (Stop on Goal)...")
    #     quantum_result = run_quantum_mcts(iterations)
    #     print("\n=== Comparison Summary (Stop on Goal) ===")
    #     print("Classical MCTS:")
    #     print(f"  Running Time: {classical_result['time']:.2f} seconds")
    #     print(f"  Solution: {classical_result['solution']}")
    #     print(f"  Goal reached at iteration: {classical_result['solution_iteration']}")
    #     
    #     print("\nQuantum MCTS:")
    #     print(f"  Running Time: {quantum_result['time']:.2f} seconds")
    #     print(f"  Solution: {quantum_result['solution']}")
    #     print(f"  Goal reached at iteration: {quantum_result['solution_iteration']}")
    # main_stop()
    
    # 2. Run until the optimal solution (known length) is found:
    main_optimal()
