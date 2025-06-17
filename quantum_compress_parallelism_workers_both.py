import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import matplotlib.pyplot as plt
import concurrent.futures

# ==========================
# Preset Mazes (including 20×20)
# ==========================
preset_mazes = [
    {"name":"2×2",
     "grid":[[0,1],
             [0,0]],
     "start":(0,0),
     "goal":(1,1)},
    {"name":"3×3",
     "grid":[[0,0,0],
             [0,1,0],
             [0,0,0]],
     "start":(0,0),
     "goal":(2,2)},
    {"name":"4×4",
     "grid":[[0,0,0,0],
             [1,1,0,1],
             [0,0,0,0],
             [0,1,1,0]],
     "start":(0,0),
     "goal":(3,3)},
    {"name":"5×5",
     "grid":[[0,0,0,0,0],
             [1,1,0,1,0],
             [0,0,0,1,0],
             [0,1,1,1,0],
             [0,0,0,0,0]],
     "start":(0,0),
     "goal":(4,4)},
    # 20×20 open maze
    {"name":"20×20",
     "grid":[[0]*20 for _ in range(20)],
     "start":(0,0),
     "goal":(19,19)},
]

# ==========================
# Maze Class
# ==========================
class Maze:
    def __init__(self, grid, start, goal):
        self.grid  = grid
        self.start = start
        self.goal  = goal
        self.rows  = len(grid)
        self.cols  = len(grid[0])
    def is_valid(self, pos):
        x,y = pos
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x]==0
    def get_moves(self, pos):
        x,y = pos
        cand = [("up",(x,y-1)),("down",(x,y+1)),
                ("left",(x-1,y)),("right",(x+1,y))]
        return [(d,p) for d,p in cand if self.is_valid(p)]
    def reached_target(self, pos):
        return pos==self.goal

# ==========================
# Helpers
# ==========================
def manhattan_distance(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# --- Classical RNG ---
def classical_choice(options):
    return random.choice(options)

# --- Quantum RNG w/ batching & cache ---
GLOBAL_QCACHE = {}
GLOBAL_BATCH  = {}
_BATCH_SIZE   = 10000

def quantum_choice(options):
    n = len(options)
    if n==1:
        return options[0]
    q = math.ceil(log2(n))
    # compile once per q
    if q not in GLOBAL_QCACHE:
        dev = qml.device("default.qubit", wires=q)
        @qml.qnode(dev)
        def circ():
            for i in range(q):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(q))
        GLOBAL_QCACHE[q] = circ
    # refill batch if empty
    if q not in GLOBAL_BATCH or not GLOBAL_BATCH[q]:
        probs = GLOBAL_QCACHE[q]()[:2**q]
        GLOBAL_BATCH[q] = list(
            np.random.choice(np.arange(2**q),
                             size=_BATCH_SIZE,
                             p=probs)
        )
    idx = GLOBAL_BATCH[q].pop()
    if idx < n:
        return options[idx]
    else:
        return quantum_choice(options)

# ==========================
# MCTS Node
# ==========================
class MCTSNode:
    __slots__ = ("pos","parent","move","children","visits","wins","untried")
    def __init__(self,pos,parent=None,move=None):
        self.pos      = pos
        self.parent   = parent
        self.move     = move
        self.children = []
        self.visits   = 0
        self.wins     = 0
        self.untried  = None
    def fully_expanded(self,maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        return len(self.untried)==0
    def expand(self,maze,selector):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = MCTSNode(mv[1], parent=self, move=mv[0])
        self.children.append(child)
        return child
    def best_child(self,c=1.4):
        best_score = -1e9
        best_node  = None
        for ch in self.children:
            if ch.visits==0:
                score = float("inf")
            else:
                score = (ch.wins/ch.visits) + \
                        c * math.sqrt(math.log(self.visits)/ch.visits)
            if score>best_score:
                best_score,best_node = score,ch
        return best_node
    def build_path(self):
        p,cur = [],self
        while cur:
            p.append(cur.pos)
            cur=cur.parent
        return p[::-1]

# ==========================
# Rollout w/ Manhattan filtering
# ==========================
def rollout(start,maze,selector,max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached_target(cur):
            return 1
        moves = maze.get_moves(cur)
        if not moves:
            break
        d0 = manhattan_distance(cur, maze.goal)
        better = [m for m in moves
                  if manhattan_distance(m[1],maze.goal)<d0]
        if better:
            moves = better
        cur = selector(moves)[1]
    return 0

# ==========================
# MCTS Search (stop on goal)
# ==========================
def mcts_stop(maze,iterations,selector):
    root = MCTSNode(maze.start)
    for i in range(1,iterations+1):
        node = root
        # selection
        while node.fully_expanded(maze) and node.children \
              and not maze.reached_target(node.pos):
            node = node.best_child()
        # expansion
        if not maze.reached_target(node.pos):
            node = node.expand(maze, selector)
        # simulation
        result = rollout(node.pos, maze, selector)
        # backprop
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += result
            tmp = tmp.parent
        # stop if goal
        if maze.reached_target(node.pos):
            return root, i
    return root, None

def extract_path(node,maze):
    if maze.reached_target(node.pos):
        return node.build_path()
    for ch in node.children:
        sol = extract_path(ch,maze)
        if sol:
            return sol
    return []

# ==========================
# Workers & Parallel Runner
# ==========================
def worker_classical(args):
    maze,its = args
    t0 = time.time()
    root,first = mcts_stop(maze, its, classical_choice)
    dt = time.time()-t0
    return {"time":dt, "first":first, "path":extract_path(root,maze)}

def worker_quantum(args):
    maze,its = args
    t0 = time.time()
    root,first = mcts_stop(maze, its, quantum_choice)
    dt = time.time()-t0
    return {"time":dt, "first":first, "path":extract_path(root,maze)}

def parallel_run(fn, maze, its, workers=12):
    tasks = [(maze,its)] * workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fn, tasks))
    return min(results, key=lambda r: r["time"])

# ==========================
# Plotting
# ==========================
def plot_maze(maze,path,title):
    plt.figure(figsize=(5,5))
    plt.imshow(np.array(maze.grid), cmap="binary", origin="upper")
    sx,sy = maze.start; gx,gy = maze.goal
    plt.scatter(sx,sy,c="green",marker="o",s=100,label="Start")
    plt.scatter(gx,gy,c="red",marker="X",s=100,label="Goal")
    if path:
        xs,ys = zip(*path)
        plt.plot(xs,ys,linewidth=2,label="Path")
        plt.scatter(xs,ys,s=50)
    plt.title(title); plt.legend(); plt.show()

# ==========================
# Main
# ==========================
if __name__=="__main__":
    iterations = 200
    workers    = 12

    for m in preset_mazes:
        print(f"\n--- Maze {m['name']} ---")
        mz = Maze(m["grid"], m["start"], m["goal"])

        c = parallel_run(worker_classical, mz, iterations, workers)
        print(f"Classical: time={c['time']:.4f}s  first_it={c['first']}  len={len(c['path'])}")
        plot_maze(mz, c['path'], title=f"Classical {m['name']}")

        q = parallel_run(worker_quantum, mz, iterations, workers)
        print(f"Quantum:   time={q['time']:.4f}s  first_it={q['first']}  len={len(q['path'])}")
        plot_maze(mz, q['path'], title=f"Quantum {m['name']}")
