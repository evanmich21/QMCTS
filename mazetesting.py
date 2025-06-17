import pennylane as qml
import numpy as np
from math import ceil, log2
import math, random

# Quantum randomness helper: returns a randomly chosen option using a quantum circuit.
def quantum_random_choice(options):
    """
    Uses a quantum circuit (with PennyLane's default.qubit simulator)
    to randomly select one element from 'options'.
    
    It builds a circuit with enough wires to represent indices 0,...,n-1,
    then samples from the uniform distribution in superposition.
    If the measured index is out of range, it retries via rejection sampling.
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

# Define a simple 3x3 Maze.
class Maze3x3:
    def __init__(self):
        # Maze grid: 0 = free cell, 1 = wall.
        # Layout:
        #   [0, 0, 0]
        #   [1, 1, 0]
        #   [0, 0, 0]
        self.grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        self.start = (0, 0)
        self.goal = (2, 2)
        self.rows = 3
        self.cols = 3

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

# MCTS Node to represent a state in the search tree.
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state          # Maze position, e.g., (x, y)
        self.parent = parent        # Parent node in the tree
        self.move = move            # Move taken from the parent to reach this state
        self.children = []          # List of child nodes
        self.visits = 0             # Number of times this node was visited
        self.wins = 0               # Cumulative reward (1 for win, -1 for loss)
        self.untried_moves = None   # Moves not yet expanded from this node

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

# Simulation (rollout) using quantum randomness.
def simulate(maze, state, max_steps=50):
    current = state
    steps = 0
    while current != maze.goal and steps < max_steps:
        moves = maze.get_moves(current)
        if not moves:
            break  # Dead end
        chosen_move = quantum_random_choice(moves)
        current = chosen_move[1]
        steps += 1
    # Return +1 if goal is reached, else -1 (a loss).
    return 1 if current == maze.goal else -1

# Back propagation: update nodes from the current node up to the root.
def backpropagate(node, result, verbose=False):
    while node is not None:
        node.visits += 1
        node.wins += result
        if verbose:
            print(f"    Node {node.state}: visits={node.visits}, wins={node.wins}")
        node = node.parent

# The main MCTS function that integrates selection, expansion, simulation, and back propagation.
def mcts(maze, iterations=1000, verbose=False):
    root = MCTSNode(maze.start)
    solution_iteration = None  # Record which iteration first found the goal.
    for i in range(iterations):
        node = root
        
        if verbose:
            print(f"Iteration {i+1} starting at root {root.state}")

        # SELECTION: traverse the tree until a node is not fully expanded or is terminal.
        while node.state != maze.goal and node.is_fully_expanded(maze) and node.children:
            node = node.best_child()
            if verbose:
                print(f"  Selection: moved to node {node.state} via move '{node.move}', visits={node.visits}, wins={node.wins}")
        
        # EXPANSION: if the node is not terminal, expand one child.
        if node.state != maze.goal:
            expanded = node.expand(maze)
            if verbose:
                print(f"  Expansion: expanded move '{expanded.move}' to state {expanded.state}")
            node = expanded
        
        # Check if we just created a node that is the goal.
        if node.state == maze.goal and solution_iteration is None:
            solution_iteration = i + 1
            print(f"*** Goal reached at iteration {solution_iteration} ***")
        
        # SIMULATION: rollout from the node.
        sim_result = simulate(maze, node.state)
        if verbose:
            print(f"  Simulation: rollout from state {node.state} resulted in reward {sim_result}")
        
        # BACK PROPAGATION: update the tree with the simulation result.
        if verbose:
            print("  Backpropagation:")
        backpropagate(node, sim_result, verbose=verbose)
    
    if solution_iteration is not None:
        print(f"Solution was first found at iteration: {solution_iteration}")
    else:
        print("No solution was found within the given iterations.")
    return root

# Helper function to extract a solution path from the MCTS tree.
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

if __name__ == "__main__":
    maze = Maze3x3()
    # Set verbose=True to see detailed iteration output.
    root = mcts(maze, iterations=50, verbose=True)
    solution = extract_solution(root, maze)
    print("\nSolution path found:")
    print(solution)
