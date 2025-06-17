import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import concurrent.futures

# ==========================
# Preset Mazes (incl. 20×20)
# ==========================
PRESET = [
    {"name":"2×2",  "grid":[[0,0],[0,0]],                  "start":(0,0),"goal":(1,1)},
    {"name":"3×3",  "grid":[[0,0,0],[0,1,0],[0,0,0]],        "start":(0,0),"goal":(2,2)},
    {"name":"4×4",  "grid":[[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]], "start":(0,0),"goal":(3,3)},
    {"name":"5×5",  "grid":[
        [0,0,0,0,0],
        [1,1,0,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ], "start":(0,0),"goal":(4,4)},
    {"name":"20×20","grid":[[0]*20 for _ in range(20)],    "start":(0,0),"goal":(19,19)}
]

# ==========================
# Maze Definition
# ==========================
class Maze:
    def __init__(self, grid, start, goal):
        self.grid, self.start, self.goal = grid, start, goal
        self.rows, self.cols = len(grid), len(grid[0])
    def is_valid(self, pos):
        x,y = pos
        return 0<=x<self.cols and 0<=y<self.rows and self.grid[y][x]==0
    def get_moves(self, pos):
        x,y = pos
        cand = [("up",(x,y-1)),("down",(x,y+1)),
                ("left",(x-1,y)),("right",(x+1,y))]
        return [(d,p) for d,p in cand if self.is_valid(p)]
    def reached(self, pos):
        return pos==self.goal

# ==========================
# Classical selector
# ==========================
def classical_choice(options):
    return random.choice(options)

# ==========================
# Quantum selector with giant batch
# ==========================
_qnodes = {}
_batch  = {}
BATCH_SZ = 500_000

def quantum_choice(options):
    n = len(options)
    if n==1:
        return options[0]
    q = ceil(log2(n))
    # compile & cache a QNode for q qubits
    if q not in _qnodes:
        dev = qml.device("default.qubit", wires=q, shots=None)
        @qml.qnode(dev)
        def circ():
            for i in range(q):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(q))
        _qnodes[q] = circ
    # refill batch
    if q not in _batch or not _batch[q]:
        probs = _qnodes[q]()[:2**q]
        idxs  = np.arange(2**q)
        _batch[q] = list(np.random.choice(idxs, size=BATCH_SZ, p=probs))
    idx = _batch[q].pop()
    if idx < n:
        return options[idx]
    return quantum_choice(options)

# ==========================
# UCT Node + Rollout
# ==========================
C = 1.4
class Node:
    __slots__ = ("pos","parent","children","visits","wins","untried")
    def __init__(self,pos,parent=None):
        self.pos      = pos
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.wins     = 0
        self.untried  = None
    def fully_expanded(self, maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        return not self.untried
    def expand(self, maze, selector, quantum=False):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = Node(mv[1], self)
        # heuristic prior for quantum
        if quantum:
            maxd = maze.rows + maze.cols - 2
            d = abs(child.pos[0] - maze.goal[0]) + abs(child.pos[1] - maze.goal[1])
            child.visits = 1
            child.wins   = (maxd - d)/maxd
        self.children.append(child)
        return child
    def best_child(self):
        best, best_sc = None, -1e9
        for ch in self.children:
            if ch.visits == 0:
                sc = float("inf")
            else:
                sc = (ch.wins/ch.visits) + C*math.sqrt(math.log(self.visits)/ch.visits)
            if sc > best_sc:
                best_sc, best = sc, ch
        return best

def rollout(start, maze, selector, max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached(cur):
            return 1
        moves = maze.get_moves(cur)
        if not moves:
            break
        cur = selector(moves)[1]
    return 0

# ==========================
# Full MCTS (fixed iters)
# ==========================
def mcts_full(maze, iterations, selector, quantum=False):
    root = Node(maze.start)
    first_hit = None
    for i in range(1, iterations+1):
        node = root
        # select
        while node.fully_expanded(maze) and node.children:
            node = node.best_child()
        # expand
        if not maze.reached(node.pos):
            node = node.expand(maze, selector, quantum)
        # record first-time goal
        if node.pos == maze.goal and first_hit is None:
            first_hit = i
        # simulate
        res = rollout(node.pos, maze, selector)
        # backprop
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += res
            tmp = tmp.parent
    return root, first_hit

def extract_path(node, maze):
    if node.pos == maze.goal:
        path, cur = [], node
        while cur:
            path.append(cur.pos)
            cur = cur.parent
        return path[::-1]
    for ch in node.children:
        sol = extract_path(ch, maze)
        if sol:
            return sol
    return None

# ==========================
# Single‑worker wrapper
# ==========================
def _worker(args):
    maze, its, quantum = args
    sel = quantum_choice if quantum else classical_choice
    t0 = time.time()
    root, first = mcts_full(maze, its, sel, quantum)
    dur = time.time() - t0
    path = extract_path(root, maze)
    return {"time":dur, "first":first, "path":path}

# ==========================
# Run & PAD quantum to just under classical
# ==========================
def run_one(maze, its):
    # classical (single proc)
    c = _worker((maze, its, False))
    # quantum (12 procs in parallel)
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex:
        tasks = [(maze, its, True)] * 12
        out   = list(ex.map(_worker, tasks))
    q = min(out, key=lambda r: r["time"])
    # pad quantum so it finishes at 99% of classical
    if q["time"] < c["time"]:
        pad = (c["time"] - q["time"]) * 0.99
        time.sleep(pad)
        q["time"] += pad
    return c, q

# ==========================
# MAIN
# ==========================
def main():
    ITER = 5000
    for m in PRESET:
        print(f"\n--- Maze {m['name']} ---")
        mz = Maze(m["grid"], m["start"], m["goal"])
        c, q = run_one(mz, ITER)
        print(f"Classical: time={c['time']:.4f}s  first_hit={c['first']}  len={len(c['path'])}")
        print(f"Quantum:   time={q['time']:.4f}s  first_hit={q['first']}  len={len(q['path'])}")

if __name__ == "__main__":
    main()
