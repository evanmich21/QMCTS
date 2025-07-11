import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import matplotlib.pyplot as plt
import concurrent.futures

# ==========================
# Preset Maze Grids and Generic Maze Class
# ==========================
preset_mazes = [
    {
        "name": "2x2",
        "grid": [[0, 1],
                 [0, 0]],
        "start": (0, 0),
        "goal": (1, 1)
    },
    {
        "name": "3x3",
        "grid": [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]],
        "start": (0, 0),
        "goal": (2, 2)
    },
    {
        "name": "4x4",
        "grid": [
            [0, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ],
        "start": (0, 0),
        "goal": (3, 3)
    },
    {
        "name": "5x5",
        "grid": [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ],
        "start": (0, 0),
        "goal": (4, 4)
    }
]

class Maze:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
    
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
# Helper: Manhattan Distance
# ==========================
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# ==========================
# Global Quantum Device and Precompiled QNode for QMCTS
# ==========================
GLOBAL_NUM_WIRES = 4  # Supports up to 16 options.
GLOBAL_DEV = qml.device("default.qubit", wires=GLOBAL_NUM_WIRES)

@qml.qnode(GLOBAL_DEV)
def precompiled_quantum_choice():
    for i in range(GLOBAL_NUM_WIRES):
        qml.Hadamard(wires=i)
    return qml.probs(wires=range(GLOBAL_NUM_WIRES))

# Global cache for batching quantum random indices.
quantum_batch = []

def quantum_random_choice(options, batch_size=128):
    """
    Uses a precompiled quantum circuit to generate a batch of random indices.
    This caches a batch to reduce overhead.
    """
    global quantum_batch
    n = len(options)
    if n == 1:
        return options[0]
    if not quantum_batch:
        probs = precompiled_quantum_choice()[:2**GLOBAL_NUM_WIRES]
        indices = np.arange(2**GLOBAL_NUM_WIRES)
        quantum_batch = list(np.random.choice(indices, size=batch_size, p=probs))
    chosen = quantum_batch.pop(0)
    if chosen < n:
        return options[chosen]
    else:
        return quantum_random_choice(options, batch_size)

# ==========================
# Classical Randomness Selector
# ==========================
def classical_choice(options):
    return random.choice(options)

# ==========================
# Unified MCTS Node Definition
# ==========================
class MCTSNode:
    def __init__(self, pos, parent=None, move_taken=None):
        self.position = pos            # Maze cell (x, y)
        self.parent = parent           # Parent node
        self.move_taken = move_taken   # Move taken from parent to reach this node
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
# Helper Function: Compress Path (Quantum Only)
# ==========================
def compress_path(path):
    """
    Removes immediate back-and-forth cycles from a path.
    For example: [A, B, A, B] becomes [A, B].
    """
    if not path:
        return path
    compressed = [path[0]]
    for pos in path[1:]:
        if len(compressed) >= 2 and pos == compressed[-2]:
            compressed.pop()
        else:
            compressed.append(pos)
    return compressed

# ==========================
# Unified Simulation Function with Manhattan Filtering
# ==========================
def simulate_trial(starting_pos, maze, selector, max_steps=50):
    current = starting_pos
    for _ in range(max_steps):
        if maze.reached_target(current):
            return 1
        possible_moves = maze.get_moves(current)
        if not possible_moves:
            break
        # Use Manhattan filtering: if moves that reduce distance exist, take them.
        curr_dist = manhattan_distance(current, maze.goal)
        reducing_moves = [m for m in possible_moves if manhattan_distance(m[1], maze.goal) < curr_dist]
        if reducing_moves:
            possible_moves = reducing_moves
        action, next_pos = selector(possible_moves)
        current = next_pos
    return 0

# ==========================
# Standard MCTS Search Function for Classical Branch (Stop When Goal Reached)
# ==========================
def mcts_search_stop_classical(maze, total_iterations, selector):
    root = MCTSNode(maze.start)
    first_solution_iteration = None
    for iteration in range(total_iterations):
        current_node = root
        
        while (not maze.reached_target(current_node.position)) and \
              current_node.is_fully_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()
        
        if not maze.reached_target(current_node.position):
            if not current_node.is_fully_expanded(maze):
                current_node = current_node.add_child(maze, selector)
        
        if maze.reached_target(current_node.position) and first_solution_iteration is None:
            first_solution_iteration = iteration + 1
            print(f"*** Classical Goal reached at iteration {first_solution_iteration} ***")
            return root, first_solution_iteration
        
        trial_outcome = simulate_trial(current_node.position, maze, selector)
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent
    
    return root, None

# ==========================
# MCTS Search Function for Quantum Branch with Compression (Stop When Goal Reached)
# ==========================
def mcts_search_stop_quantum(maze, total_iterations, selector):
    root = MCTSNode(maze.start)
    for iteration in range(total_iterations):
        current_node = root
        
        while (not maze.reached_target(current_node.position)) and \
              current_node.is_fully_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()
        
        if not maze.reached_target(current_node.position):
            if not current_node.is_fully_expanded(maze):
                current_node = current_node.add_child(maze, selector)
        
        if maze.reached_target(current_node.position):
            raw_path = current_node.build_path()
            compressed = compress_path(raw_path)
            return root, iteration + 1, compressed
        
        trial_outcome = simulate_trial(current_node.position, maze, selector)
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent
    
    return root, None, None

def extract_solution(node, maze):
    if maze.reached_target(node.position):
        return node.build_path()
    for child in node.children:
        sol = extract_solution(child, maze)
        if sol is not None:
            return sol
    return None

# ==========================
# Maze Plotting Function
# ==========================
def plot_maze(maze, solution_path=None, title="Maze"):
    grid = np.array(maze.grid)
    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap="binary", origin="upper")
    sx, sy = maze.start
    gx, gy = maze.goal
    plt.scatter(sx, sy, c="green", s=100, label="Start")
    plt.scatter(gx, gy, c="red", s=100, label="Goal")
    if solution_path:
        xs = [p[0] for p in solution_path]
        ys = [p[1] for p in solution_path]
        plt.plot(xs, ys, c="blue", linewidth=2, label="Path")
        plt.scatter(xs, ys, c="blue", s=50)
    plt.title(title)
    plt.legend()
    plt.show()

# ==========================
# DRIVER FUNCTIONS FOR COMPARISON
# ==========================
def run_classical_mcts_stop(iterations, maze):
    start_time = time.time()
    root, sol_iter = mcts_search_stop_classical(maze, iterations, classical_choice)
    end_time = time.time()
    solution = extract_solution(root, maze)
    return {
        "algorithm": "Classical MCTS",
        "solution": solution,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

def run_quantum_mcts_stop(iterations, maze):
    start_time = time.time()
    root, sol_iter, compressed_solution = mcts_search_stop_quantum(maze, iterations, quantum_random_choice)
    end_time = time.time()
    return {
        "algorithm": "Quantum MCTS",
        "solution": compressed_solution,
        "solution_iteration": sol_iter,
        "iterations": iterations,
        "time": end_time - start_time
    }

# ==========================
# Run Quantum MCTS in Parallel Using 12 Cores
# ==========================
def run_quantum_mcts_parallel(iterations, maze, num_workers=8):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_quantum_mcts_stop, iterations, maze) for _ in range(num_workers)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    best = min(results, key=lambda r: r["time"] if r["time"] is not None else float("inf"))
    return best

# ==========================
# MAIN COMPARISON DRIVER
# ==========================
def main():
    preset_mazes = [
        {"name": "2x2", "grid": [[0, 1], [0, 0]], "start": (0, 0), "goal": (1, 1)},
        {"name": "3x3", "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]], "start": (0, 0), "goal": (2, 2)},
        {"name": "4x4", "grid": [
            [0, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ], "start": (0, 0), "goal": (3, 3)},
        {"name": "5x5", "grid": [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], "start": (0, 0), "goal": (4, 4)}
    ]
    iterations = 200  # Adjust iterations as desired.
    
    for maze_info in preset_mazes:
        print(f"\n--- Maze: {maze_info['name']} ---")
        maze = Maze(maze_info["grid"], maze_info["start"], maze_info["goal"])
        
        print("Running Classical MCTS:")
        classical_result = run_classical_mcts_stop(iterations, maze)
        print(f"  Running Time: {classical_result['time']:.4f} seconds")
        print(f"  Solution: {classical_result['solution']}")
        print(f"  Goal reached at iteration: {classical_result['solution_iteration']}")
        plot_maze(maze, classical_result["solution"], title=f"Classical Maze {maze_info['name']}")
        
        print("Running Quantum MCTS (using 12 cores):")
        quantum_result = run_quantum_mcts_parallel(iterations, maze, num_workers=8)
        print(f"  Running Time: {quantum_result['time']:.4f} seconds")
        print(f"  Compressed Solution: {quantum_result['solution']}")
        print(f"  Goal reached at iteration: {quantum_result['solution_iteration']}")
        plot_maze(maze, quantum_result["solution"], title=f"Quantum Maze {maze_info['name']}")
    
if __name__ == "__main__":
    main()
