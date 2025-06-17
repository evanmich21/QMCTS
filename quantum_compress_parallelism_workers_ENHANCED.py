import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import matplotlib.pyplot as plt
import concurrent.futures

# ==========================
# Preset Mazes
# ==========================
preset_mazes = [
    {"name": "2x2", "grid": [[0,1],[0,0]], "start": (0,0), "goal": (1,1)},
    {"name": "3x3", "grid": [[0,0,0],[0,1,0],[0,0,0]], "start": (0,0), "goal": (2,2)},
    {"name": "4x4", "grid": [[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]], "start": (0,0), "goal": (3,3)},
    {"name": "5x5", "grid": [
        [0,0,0,0,0],
        [1,1,0,1,1],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ], "start": (0,0), "goal": (4,4)}
]

# ==========================
# Maze Class
# ==========================
class Maze:
    def __init__(self, grid, start, goal):
        self.grid, self.start, self.goal = grid, start, goal
        self.rows, self.cols = len(grid), len(grid[0])
    def is_valid(self, pos):
        x,y = pos
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x]==0
    def get_moves(self, pos):
        x,y = pos
        dirs = {"up":(x,y-1),"down":(x,y+1),"left":(x-1,y),"right":(x+1,y)}
        return [(d,new) for d,new in dirs.items() if self.is_valid(new)]
    def reached_target(self, pos):
        return pos==self.goal

# ==========================
# Helpers
# ==========================
def manhattan_distance(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# --------------------------
# Classical RNG
# --------------------------
def classical_choice(options):
    return random.choice(options)

# --------------------------
# Quantum RNG: analytic + cache
# --------------------------
_qcircuit_cache = {}
_probs_cache    = {}

def quantum_random_choice(options):
    n = len(options)
    if n == 1:
        return options[0]
    num_qubits = ceil(log2(n))
    # compile and cache the QNode
    if num_qubits not in _qcircuit_cache:
        dev = qml.device("lightning.qubit", wires=num_qubits, shots=None)
        @qml.qnode(dev)
        def circ():
            for i in range(num_qubits):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(num_qubits))
        _qcircuit_cache[num_qubits] = circ
    # get or compute the probability vector
    if num_qubits not in _probs_cache:
        _probs_cache[num_qubits] = _qcircuit_cache[num_qubits]()
    probs = _probs_cache[num_qubits]
    # sample via numpy
    idx = np.random.choice(len(probs), p=probs)
    return options[idx] if idx < n else quantum_random_choice(options)

# ==========================
# MCTS Node
# ==========================
class MCTSNode:
    def __init__(self,pos,parent=None,move=None):
        self.position, self.parent, self.move = pos, parent, move
        self.children = []
        self.visits = 0
        self.wins   = 0
        self.untried = None
    def build_path(self):
        path, cur = [], self
        while cur:
            path.append(cur.position)
            cur = cur.parent
        return path[::-1]
    def is_fully_expanded(self,maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.position).copy()
        return len(self.untried)==0
    def expand(self,maze,selector):
        if self.untried is None:
            self.untried = maze.get_moves(self.position).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = MCTSNode(mv[1], parent=self, move=mv[0])
        self.children.append(child)
        return child
    def best_child(self,c=1.4):
        best, best_score = None, -1e9
        for ch in self.children:
            if ch.visits==0:
                score = float("inf")
            else:
                score = (ch.wins/ch.visits) + c*math.sqrt(math.log(self.visits)/ch.visits)
            if score>best_score:
                best_score, best = score, ch
        return best

# ==========================
# Simulation (with Manhattan filtering)
# ==========================
def simulate(start,maze,selector,max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached_target(cur):
            return 1
        moves = maze.get_moves(cur)
        if not moves:
            break
        md = manhattan_distance(cur, maze.goal)
        better = [m for m in moves if manhattan_distance(m[1], maze.goal)<md]
        if better:
            moves = better
        cur = selector(moves)[1]
    return 0

# ==========================
# MCTS Search (stop when goal found)
# ==========================
def mcts_search_stop(maze,iterations,selector):
    root = MCTSNode(maze.start)
    for it in range(iterations):
        node = root
        # selection
        while (not maze.reached_target(node.position)
               and node.is_fully_expanded(maze)
               and node.children):
            node = node.best_child()
        # expansion
        if not maze.reached_target(node.position):
            node = node.expand(maze,selector)
        # check goal
        if maze.reached_target(node.position):
            return root, it+1
        # simulation
        result = simulate(node.position,maze,selector)
        # backprop
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += result
            tmp = tmp.parent
    return root, None

def extract_solution(root,maze):
    if maze.reached_target(root.position):
        return root.build_path()
    for c in root.children:
        sol = extract_solution(c,maze)
        if sol:
            return sol
    return None

# ==========================
# Parallel worker functions
# ==========================
def _run_classical(args):
    maze,iters = args
    start = time.time()
    root, sol_it = mcts_search_stop(maze,iters,classical_choice)
    return {"time":time.time()-start, "sol_it":sol_it, "path":extract_solution(root,maze)}

def _run_quantum(args):
    maze,iters = args
    start = time.time()
    root, sol_it = mcts_search_stop(maze,iters,quantum_random_choice)
    return {"time":time.time()-start, "sol_it":sol_it, "path":extract_solution(root,maze)}

def parallel_run(fn, maze, iters, workers=8):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fn, [(maze,iters)]*workers))
    return min(results, key=lambda r: r["time"])

# ==========================
# Plotting
# ==========================
def plot_maze(maze,path,title):
    plt.figure(figsize=(5,5))
    plt.imshow(np.array(maze.grid),cmap="binary",origin="upper")
    sx,sy = maze.start; gx,gy = maze.goal
    plt.scatter(sx,sy,c="green",marker="o",label="Start")
    plt.scatter(gx,gy,c="red",marker="X",label="Goal")
    if path:
        xs,ys = zip(*path)
        plt.plot(xs,ys,c="blue",linewidth=2)
        plt.scatter(xs,ys,c="blue",s=50)
    plt.title(title); plt.legend(); plt.show()

# ==========================
# Main comparison
# ==========================
def main():
    iterations = 200
    for m in preset_mazes:
        print(f"\n--- Maze {m['name']} ---")
        maze = Maze(m["grid"], m["start"], m["goal"])
        # classical
        res_c = parallel_run(_run_classical, maze, iterations, workers=8)
        print(f"Classical   time={res_c['time']:.6f}s iters={res_c['sol_it']} path={res_c['path']}")
        plot_maze(maze, res_c["path"], title=f"Classical {m['name']}")
        # quantum
        res_q = parallel_run(_run_quantum, maze, iterations, workers=8)
        print(f"Quantum     time={res_q['time']:.6f}s iters={res_q['sol_it']} path={res_q['path']}")
        plot_maze(maze, res_q["path"], title=f"Quantum {m['name']}")

if __name__=="__main__":
    main()
