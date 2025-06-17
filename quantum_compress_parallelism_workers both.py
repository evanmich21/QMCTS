import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import concurrent.futures

# ==========================
# Preset Mazes (2×2 → 5×5)
# ==========================
PRESET = [
    {"name":"2×2", "grid":[[0,0],[0,0]],                  "start":(0,0),"goal":(1,1)},
    {"name":"3×3", "grid":[[0,0,0],[0,1,0],[0,0,0]],        "start":(0,0),"goal":(2,2)},
    {"name":"4×4", "grid":[[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]], "start":(0,0),"goal":(3,3)},
    {"name":"5×5", "grid":[
        [0,0,0,0,0],
        [1,1,0,1,0],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ], "start":(0,0),"goal":(4,4)}
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
        cand = [("U",(x,y-1)),("D",(x,y+1)),("L",(x-1,y)),("R",(x+1,y))]
        return [(d,p) for d,p in cand if self.is_valid(p)]
    def reached(self, pos):
        return pos==self.goal

# ==========================
# Selectors
# ==========================
def classical_choice(options):
    return random.choice(options)

_qnodes = {}
_batch  = {}
BATCH_SZ = 200_000

def quantum_choice(options):
    n = len(options)
    if n==1:
        return options[0]
    q = ceil(log2(n))
    if q not in _qnodes:
        dev = qml.device("default.qubit", wires=q, shots=None)
        @qml.qnode(dev)
        def circ():
            for i in range(q):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(q))
        _qnodes[q] = circ
    if q not in _batch or not _batch[q]:
        probs = _qnodes[q]()[:2**q]
        idxs  = np.arange(2**q)
        _batch[q] = list(np.random.choice(idxs, size=BATCH_SZ, p=probs))
    idx = _batch[q].pop()
    return options[idx] if idx < n else quantum_choice(options)

# ==========================
# UCT Node & Rollout
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
    def fully(self, maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        return not self.untried
    def expand(self, maze, selector, quantum=False):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = Node(mv[1], self)
        if quantum:
            # heuristic prior
            maxd = maze.rows + maze.cols - 2
            d    = abs(child.pos[0]-maze.goal[0]) + abs(child.pos[1]-maze.goal[1])
            child.visits = 1
            child.wins   = (maxd - d)/maxd
        self.children.append(child)
        return child
    def best(self):
        best, best_sc = None, -1e9
        for ch in self.children:
            if ch.visits==0:
                sc = float("inf")
            else:
                sc = (ch.wins/ch.visits) + C*math.sqrt(math.log(self.visits)/ch.visits)
            if sc>best_sc:
                best_sc, best = sc, ch
        return best

def rollout(start, maze, selector, max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached(cur):
            return 1
        mv = selector(maze.get_moves(cur))
        cur = mv[1]
    return 0

# ==========================
# Full MCTS (fixed its)
# ==========================
def mcts_full(maze, its, selector, quantum=False):
    root = Node(maze.start)
    first = None
    for i in range(1, its+1):
        node = root
        while node.fully(maze) and node.children:
            node = node.best()
        if not maze.reached(node.pos):
            node = node.expand(maze, selector, quantum)
        if node.pos==maze.goal and first is None:
            first = i
        res = rollout(node.pos, maze, selector)
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += res
            tmp = tmp.parent
    return root, first

def extract_path(node, maze):
    if node.pos==maze.goal:
        path, cur = [], node
        while cur:
            path.append(cur.pos)
            cur=cur.parent
        return path[::-1]
    for ch in node.children:
        p = extract_path(ch, maze)
        if p:
            return p
    return None

# ==========================
# Worker + Parallel
# ==========================
def _worker(args):
    maze, its, quantum = args
    sel = quantum_choice if quantum else classical_choice
    t0 = time.time()
    root, first = mcts_full(maze, its, sel, quantum)
    dur = time.time() - t0
    path = extract_path(root, maze)
    return {"time":dur, "first":first, "length":len(path)}

def parallel_mcts(maze, its, quantum, workers=12):
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        tasks = [(maze, its, quantum)] * workers
        out   = list(ex.map(_worker, tasks))
    return min(out, key=lambda r:r["time"])

# ==========================
# MAIN
# ==========================
def main():
    ITS = 3000
    fmt = "{:<8}  {:>7}  {:>9}  {:>6}"
    print(fmt.format("Maze","Alg","Time(s)","FirstIt","Len"))
    print("-"*36)
    for m in PRESET:
        mz = Maze(m["grid"], m["start"], m["goal"])
        c = parallel_mcts(mz, ITS, False, workers=12)
        q = parallel_mcts(mz, ITS, True,  workers=12)
        print(fmt.format(m["name"], "Class", f"{c['time']:.4f}", c["first"], c["length"]))
        print(fmt.format("",       "Quantum",f"{q['time']:.4f}", q["first"], q["length"]))
        print()

if __name__=="__main__":
    main()
