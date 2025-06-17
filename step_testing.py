import pennylane as qml
import numpy as np
from math import ceil, log2
import math, random, time, os, concurrent.futures

# -------------------------------
# Quantum Randomness Helper
# -------------------------------
def quantum_random_choice(options):
    """
    Uses a quantum circuit (with PennyLane's default.qubit simulator)
    to randomly select one element from 'options'.
    Builds a circuit with enough wires to represent indices 0,...,n-1,
    then samples from the uniform distribution in superposition.
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

# -------------------------------
# Helper: Manhattan Distance
# -------------------------------
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# -------------------------------
# Maze Definition: 5x5 Maze
# -------------------------------
class Maze5x5:
    def __init__(self):
        # Maze grid: 0 = free cell, 1 = wall.
        # Example layout:
        #   [0, 0, 0, 0, 0]
        #   [1, 1, 0, 1, 0]
        #   [0, 0, 0, 1, 0]
        #   [0, 1, 1, 1, 0]
        #   [0, 0, 0, 0, 0]
        self.grid = [
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]
        self.start = (0, 0)
        self.goal = (4, 4)
        self.rows = 5
        self.cols = 5

    def is_valid(self, pos):
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        return self.grid[y][x] == 0

    def get_moves(self, pos):
        """
        Returns a list of valid moves from position `pos`.
        Each move is a tuple: (direction, new_position).
        Allowed directions: up, down, left, right.
        """
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

# -------------------------------
# MCTS Node Definition
# -------------------------------
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state          # Maze position (e.g., (x, y))
        self.parent = parent        # Parent node in the tree
        self.move = move            # Move taken from parent to reach this state
        self.children = []          # List of child nodes
        self.visits = 0             # Visit count
        self.wins = 0               # Cumulative reward
        self.untried_moves = None   # Moves not yet expanded

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
            child = MCTSNode(chosen_move[1], parent=self, move=chosen_move[0])
            self.children.append(child)
            return child
        return None

    def best_child(self, c_param=1.4):
        best = None
        best_score = -float('inf')
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                score = (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

# -------------------------------
# Simulation (Rollout) with Step Penalty, Repeated-State Penalty, and No Immediate Backtracking
# -------------------------------
def simulate(maze, state, max_steps=100):
    current = state
    steps = 0
    visited = {}  # Track visits for penalties.
    path = [current]
    while current != maze.goal and steps < max_steps:
        moves = maze.get_moves(current)
        
        # Extra filtering: if there are moves that reduce the Manhattan distance,
        # then filter out moves that do not reduce it.
        current_dist = manhattan_distance(current, maze.goal)
        reducing_moves = [m for m in moves if manhattan_distance(m[1], maze.goal) < current_dist]
        if reducing_moves:
            moves = reducing_moves

        # Also, disallow immediate backtracking if possible.
        if len(path) >= 2:
            prev_state = path[-2]
            non_back_moves = [m for m in moves if m[1] != prev_state]
            if non_back_moves:
                moves = non_back_moves
        
        if not moves:
            break  # Dead end.
        
        chosen_move = quantum_random_choice(moves)
        current = chosen_move[1]
        path.append(current)
        steps += 1
        visited[current] = visited.get(current, 0) + 1

    # For a 5x5 maze, maximum Manhattan distance = 8.
    max_dist = maze.rows + maze.cols - 2
    dist = manhattan_distance(current, maze.goal)
    if current == maze.goal:
        reward = 10  # Big reward for reaching the goal; no penalty.
    else:
        base_reward = (max_dist - dist) / max_dist - 0.5
        penalty = sum((count - 1) * 0.2 for count in visited.values() if count > 1)
        reward = base_reward - penalty
    # Apply a step penalty: subtract 0.1 per step taken.
    step_penalty = 0.1 * steps
    final_reward = reward - step_penalty
    return final_reward

# -------------------------------
# Backpropagation
# -------------------------------
def backpropagate(node, result, verbose=False):
    while node is not None:
        node.visits += 1
        node.wins += result
        if verbose:
            print(f"    Node {node.state}: visits={node.visits}, wins={node.wins}")
        node = node.parent

# -------------------------------
# MCTS Main Function with Best Solution Tracking and Detailed Print Summary
# -------------------------------
def mcts(maze, iterations=5000, verbose=False):
    root = MCTSNode(maze.start)
    best_solution_steps = None
    best_solution_iteration = None
    solution_iteration = None
    for i in range(iterations):
        node = root
        
        if verbose:
            print(f"Iteration {i+1} starting at root {root.state}")
        
        # Selection: descend until reaching a node not fully expanded.
        while node.state != maze.goal and node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
            if verbose:
                print(f"  Selection: moved to node {node.state} via move '{node.move}', visits={node.visits}, wins={node.wins}")
        
        # Expansion: if node is not at goal, expand one child.
        if node.state != maze.goal:
            expanded = node.expand(maze)
            if verbose:
                print(f"  Expansion: expanded move '{expanded.move}' to state {expanded.state}")
            node = expanded
        
        # If a goal node is created, record the solution.
        if node.state == maze.goal:
            path = extract_solution(node, maze)
            steps_taken = len(path)
            if best_solution_steps is None or steps_taken < best_solution_steps:
                best_solution_steps = steps_taken
                best_solution_iteration = i + 1
                print(f"*** New best solution found at iteration {best_solution_iteration} with {steps_taken} steps ***")
            if solution_iteration is None:
                solution_iteration = i + 1
                print(f"*** Goal reached at iteration {solution_iteration} ***")
        
        sim_result = simulate(maze, node.state)
        if verbose:
            print(f"  Simulation: rollout from state {node.state} resulted in reward {sim_result}")
        
        if verbose:
            print("  Backpropagation:")
        backpropagate(node, sim_result, verbose=verbose)
    
    # Print final summary.
    if solution_iteration is not None:
        print(f"\nSolution was first found at iteration: {solution_iteration}")
        print(f"Best (shortest) solution was found at iteration {best_solution_iteration} with {best_solution_steps} steps")
    else:
        print("\nNo solution was found within the given iterations.")
    return root, solution_iteration, best_solution_iteration, best_solution_steps

# -------------------------------
# Helper: Extract Solution Path
# -------------------------------
def extract_solution(node, maze):
    if node.state == maze.goal:
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]
    for child in node.children:
        sol = extract_solution(child, maze)
        if sol is not None:
            return sol
    return None

# -------------------------------
# Function to Run MCTS (for Parallel Execution)
# -------------------------------
def run_mcts(instance_id, iterations=5000):
    maze = Maze5x5()
    root, sol_iter, best_iter, best_steps = mcts(maze, iterations=iterations, verbose=True)
    solution = extract_solution(root, maze)
    return {
        "instance_id": instance_id,
        "solution": solution,
        "solution_iteration": sol_iter,
        "best_solution_iteration": best_iter,
        "best_solution_steps": best_steps
    }

# -------------------------------
# Main: Parallel Execution with 10 Tasks
# -------------------------------
if __name__ == "__main__":
    num_instances = 10  # Fixed 10 parallel tasks.
    iterations_per_instance = 5000

    print(f"Running {num_instances} parallel MCTS instances on a 5x5 maze...")
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_mcts, i, iterations_per_instance) for i in range(num_instances)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    end_time = time.time()

    print("\nSummary of Parallel Runs:")
    for res in sorted(results, key=lambda x: x["instance_id"]):
        print(f"Instance {res['instance_id']}:")
        print(f"  Solution path: {res['solution']}")
        print(f"  Goal first reached at iteration: {res['solution_iteration']}")
        print(f"  Best (shortest) solution found at iteration: {res['best_solution_iteration']} with {res['best_solution_steps']} steps")
    
    # Determine the best overall solution across all instances.
    best_overall = None
    for res in results:
        if res["solution"] is not None:
            if best_overall is None or res["best_solution_steps"] < best_overall["best_solution_steps"]:
                best_overall = res

    if best_overall is not None:
        print("\nBest overall solution:")
        print(f"  Instance: {best_overall['instance_id']}")
        print(f"  Solution path: {best_overall['solution']}")
        print(f"  Found at iteration: {best_overall['best_solution_iteration']} with {best_overall['best_solution_steps']} steps")
    else:
        print("\nNo solution was found in any instance.")
    
    print(f"\nTotal parallel execution time: {end_time - start_time:.2f} seconds")
