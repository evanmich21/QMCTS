import math
import random
import numpy as np
import matplotlib.pyplot as plt

# --- Maze Definition ---

class MazePuzzle:
    def __init__(self):
        self.rows = 4
        self.cols = 4
        # Grid layout: 0 = open cell, 1 = wall.
        self.grid_map = [
            [0, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0]
        ]
        self.start_pos = (0, 0)
        self.target_pos = (3, 3)

    def is_move_allowed(self, pos):
        x, y = pos
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return False
        if self.grid_map[y][x] == 1:
            return False
        return True

    def list_moves(self, pos):
        moves_list = []
        x, y = pos
        # Directions: North, South, West, East.
        directions = {
            'N': (x, y - 1),
            'S': (x, y + 1),
            'W': (x - 1, y),
            'E': (x + 1, y)
        }
        for direction, new_pos in directions.items():
            if self.is_move_allowed(new_pos):
                moves_list.append((direction, new_pos))
        return moves_list

    def reached_target(self, pos):
        return pos == self.target_pos

# --- Tree Node Definition ---

class TreeNode:
    def __init__(self, pos, parent=None, move_taken=None):
        self.position = pos             # Maze cell (x, y)
        self.parent = parent            # Reference to parent node
        self.move_taken = move_taken    # Action used to reach this node
        self.children = []              # List of child nodes
        self.visit_count = 0            # Number of visits
        self.win_count = 0              # Successful simulation count from this node
        self.available_actions = None   # Moves yet to try from this node

    def build_path(self):
        """Returns the sequence of positions from the start to this node."""
        path = []
        curr = self
        while curr:
            path.append(curr.position)
            curr = curr.parent
        return path[::-1]

    def is_completely_expanded(self, maze):
        if self.available_actions is None:
            self.available_actions = maze.list_moves(self.position).copy()
        return len(self.available_actions) == 0

    def add_child(self, maze):
        if self.available_actions is None:
            self.available_actions = maze.list_moves(self.position).copy()
        if self.available_actions:
            idx = random.randrange(len(self.available_actions))
            action, new_pos = self.available_actions.pop(idx)
            child_node = TreeNode(new_pos, parent=self, move_taken=action)
            self.children.append(child_node)
            return child_node
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

# --- Simulation (Rollout) Function ---

def simulate_trial(starting_pos, maze, max_steps=50):
    current = starting_pos
    for _ in range(max_steps):
        if maze.reached_target(current):
            return 1  # Success
        possible_moves = maze.list_moves(current)
        if not possible_moves:
            break  # Dead end reached
        action, next_pos = random.choice(possible_moves)
        current = next_pos
    return 0  # Trial failed

# --- Monte Carlo Tree Exploration Function ---

def mct_search(maze, total_iterations):
    root_node = TreeNode(maze.start_pos)
    solution_iteration = None

    for iteration in range(total_iterations):
        print(f"\nIteration {iteration + 1}")
        current_node = root_node

        # Selection: Follow the tree until a node that can be expanded is found.
        while not maze.reached_target(current_node.position) and \
              current_node.is_completely_expanded(maze) and current_node.children:
            current_node = current_node.choose_child()
            print(f"  [Select] {current_node.move_taken} -> {current_node.position} "
                  f"(Visits: {current_node.visit_count}, Wins: {current_node.win_count})")

        # Expansion: If current node is not terminal, add a new child.
        if not maze.reached_target(current_node.position):
            if not current_node.is_completely_expanded(maze):
                current_node = current_node.add_child(maze)
                print(f"  [Expand] New branch via {current_node.move_taken} to {current_node.position}")
            else:
                print("  [Expand] Node fully expanded; no new branch added.")

        # Record the iteration if the target is reached for the first time.
        if maze.reached_target(current_node.position) and solution_iteration is None:
            solution_iteration = iteration + 1
            print(f"*** Goal reached at iteration {solution_iteration} ***")

        # Simulation: Run a trial from the current node's position.
        trial_outcome = simulate_trial(current_node.position, maze)
        print(f"  [Simulate] Trial from {current_node.position} resulted in "
              f"{'SUCCESS' if trial_outcome == 1 else 'FAILURE'}.")

        # Backpropagation: Update stats along the path.
        node_pointer = current_node
        while node_pointer is not None:
            node_pointer.visit_count += 1
            if trial_outcome == 1:
                node_pointer.win_count += 1
            node_pointer = node_pointer.parent

    return root_node, solution_iteration

# --- Retrieve the Solution Path ---

def extract_solution(node, maze):
    """Recursively searches for a node that reached the target and returns the path."""
    if maze.reached_target(node.position):
        return node.build_path()
    for child in node.children:
        sol = extract_solution(child, maze)
        if sol is not None:
            return sol
    return None

# --- Maze Visualization ---

def plot_maze(maze, solution_path=None):
    grid = np.array(maze.grid_map)
    fig, ax = plt.subplots(figsize=(5, 5))
    # Use a binary colormap (walls: black, free cells: white)
    ax.imshow(grid, cmap="binary", origin="upper")

    # Mark starting and target cells with the original colors.
    sx, sy = maze.start_pos
    tx, ty = maze.target_pos
    ax.scatter(sx, sy, c="green", s=200, marker="o", label="Start")
    ax.scatter(tx, ty, c="red", s=200, marker="x", label="Goal")

    # If a solution path exists, plot it using blue.
    if solution_path:
        xs = [cell[0] for cell in solution_path]
        ys = [cell[1] for cell in solution_path]
        ax.plot(xs, ys, color="blue", linewidth=2, label="Path")
        ax.scatter(xs, ys, color="blue", s=50)

    # Create gridlines for clarity.
    ax.set_xticks(np.arange(-0.5, maze.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, maze.cols - 0.5)
    ax.set_ylim(maze.rows - 0.5, -0.5)

    ax.legend(loc="upper right")
    plt.title("Maze Exploration and Solution")
    plt.show()

# --- Main Execution ---

def main():
    puzzle = MazePuzzle()
    iterations = 200  # Adjust the number of iterations if desired.
    print("Starting Monte Carlo Tree Exploration for the Maze Puzzle...\n")
    
    root, solved_iter = mct_search(puzzle, iterations)
    solution = extract_solution(root, puzzle)
    
    if solution:
        print("\nSolution path found:")
        for pos in solution:
            print(pos)
        if solved_iter:
            print(f"\nThe target was first reached on iteration {solved_iter}.")
    else:
        print("\nNo solution path was discovered.")
    
    plot_maze(puzzle, solution)

if __name__ == "__main__":
    main()
