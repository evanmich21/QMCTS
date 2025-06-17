import random
import time

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state):
        child = MCTSNode(child_state, parent=self)
        self.children.append(child)
        return child

def mcts_search(root, env, iterations=100):
    for i in range(iterations):
        node = selection(root)
        reward = simulation(node, env)
        backpropagation(node, reward)
        print(f"Iteration {i+1}: Current node state: {node.state}, Reward: {reward}")

    print("MCTS completed")
    best_path = get_best_path(root)
    return best_path

def selection(node):
    while node.children:
        node = max(node.children, key=lambda n: uct_value(n))
    return node

def simulation(node, env):
    env.reset()
    env.current_pos = node.state  # Set the environment state to the node's state
    reward = 0
    path = [node.state]
    for _ in range(10):  # Limit the number of random moves in the simulation
        action = random.choice(["up", "down", "left", "right"])
        new_state, done = env.step(action)
        path.append(new_state)
        if done:
            reward = 1  # Reached goal
            break
    return reward

def backpropagation(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def uct_value(node, exploration=1.41):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + exploration * (2 * (node.parent.visits ** 0.5) / node.visits)

def get_best_path(node):
    path = []
    while node.children:
        node = max(node.children, key=lambda n: n.visits)
        path.append(node.state)
    return path