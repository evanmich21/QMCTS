import numpy as np
import matplotlib.pyplot as plt

class MazeEnv:
    def __init__(self, maze, start=(0, 0), goal=None):
        self.maze = np.array(maze)
        self.start = start
        self.goal = goal if goal else (len(maze) - 1, len(maze[0]) - 1)
        self.current_pos = self.start

    def reset(self):
        self.current_pos = self.start
        return self.current_pos

    def step(self, action):
        moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        new_pos = (self.current_pos[0] + moves[action][0], self.current_pos[1] + moves[action][1])

        # Check if the move is valid
        if 0 <= new_pos[0] < self.maze.shape[0] and 0 <= new_pos[1] < self.maze.shape[1]:
            if self.maze[new_pos] == 0:  # 0 indicates a walkable path
                self.current_pos = new_pos

        done = self.current_pos == self.goal
        return self.current_pos, done

    def render(self, path=None):
        maze_copy = self.maze.copy()
        if path:
            for pos in path:
                if pos != self.start and pos != self.goal:
                    maze_copy[pos] = 0.5  # Mark path positions
        maze_copy[self.start] = 0.3  # Start position marker
        maze_copy[self.goal] = 0.7  # Goal position marker
        maze_copy[self.current_pos] = 2  # Current position marker

        plt.imshow(maze_copy, cmap='viridis', interpolation='nearest')
        plt.title("Maze Navigation")
        plt.colorbar(label='Path Legend')
        plt.show()