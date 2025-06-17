from maze_env import MazeEnv
from MCTS import MCTSNode, mcts_search

# Define the maze (0 = path, 1 = wall)
maze = [
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0]
]

# Initialize environment and root node
env = MazeEnv(maze)
root = MCTSNode(env.start)

# Run MCTS search
mcts_search(root, env, iterations=100)

# Render the maze
env.render()
