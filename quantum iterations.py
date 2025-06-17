import math
import random
import numpy as np
import time
from math import ceil, log2
import pennylane as qml
import concurrent.futures

# ==========================
# Mazes to test
# ==========================
preset_mazes = [
    {"name":"2×2", "grid":[[0,1],[0,0]], "start":(0,0), "goal":(1,1)},
    {"name":"3×3", "grid":[[0,0,0],[0,1,0],[0,0,0]], "start":(0,0),"goal":(2,2)},
    {"name":"4×4", "grid":[[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]], "start":(0,0),"goal":(3,3)},
    {"name":"5×5", "grid":[
        [0,0,0,0,0],
        [1,1,0,1,1],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ], "start":(0,0),"goal":(4,4)}
]

# ==========================
# Maze class
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
        return 0<=x<self.cols and 0<=y<self.rows and self.grid[y][x]==0
    def get_moves(self, pos):
        x,y = pos
        cand = [("up",(x,y-1)),("down",(x,y+1)),
                ("left",(x-1,y)),("right",(x+1,y))]
        return [(d,p) for d,p in cand if self.is_valid(p)]
    def reached_target(self,pos):
        return pos==self.goal

# ==========================
# Helpers
# ==========================
def manhattan_distance(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def classical_choice(options):
    return random.choice(options)

# --- Quantum RNG with batching ---
_GLOBAL_QNODES = {}
_GLOBAL_BATCH  = {}
_BATCH_SIZE    = 1000

def quantum_choice(options):
    n = len(options)
    if n==1:
        return options[0]
    q = math.ceil(log2(n))
    # compile QNode once per wire-count q
    if q not in _GLOBAL_QNODES:
        dev = qml.device("default.qubit", wires=q)
        @qml.qnode(dev)
        def circ():
            for i in range(q):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(q))
        _GLOBAL_QNODES[q] = circ
    # refill batch if empty
    if q not in _GLOBAL_BATCH or not _GLOBAL_BATCH[q]:
        probs = _GLOBAL_QNODES[q]()[:2**q]
        _GLOBAL_BATCH[q] = list(np.random.choice(
            np.arange(2**q), size=_BATCH_SIZE, p=probs
        ))
    idx = _GLOBAL_BATCH[q].pop()
    return options[idx] if idx < n else quantum_choice(options)

# ==========================
# MCTS core
# ==========================
class MCTSNode:
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
        return len(self.untried)==0
    def expand(self, maze, selector):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = MCTSNode(mv[1], parent=self)
        self.children.append(child)
        return child
    def best_child(self, c=1.4):
        best, best_score = None, -1e9
        for ch in self.children:
            if ch.visits==0:
                score = float("inf")
            else:
                score = (ch.wins/ch.visits) + c * math.sqrt(math.log(self.visits)/ch.visits)
            if score>best_score:
                best_score, best = score, ch
        return best
    def build_path(self):
        path, cur = [], self
        while cur:
            path.append(cur.pos)
            cur = cur.parent
        return list(reversed(path))

def rollout(start, maze, selector, max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached_target(cur):
            return 1
        moves = maze.get_moves(cur)
        if not moves:
            break
        # Manhattan‐heuristic bias
        d0 = manhattan_distance(cur, maze.goal)
        better = [m for m in moves if manhattan_distance(m[1],maze.goal)<d0]
        if better:
            moves = better
        cur = selector(moves)[1]
    return 0

def mcts_stop(maze, iterations, selector):
    root = MCTSNode(maze.start)
    for i in range(1, iterations+1):
        node = root
        # selection
        while node.fully_expanded(maze) and node.children and not maze.reached_target(node.pos):
            node = node.best_child()
        # expansion
        if not maze.reached_target(node.pos):
            node = node.expand(maze, selector)
        # simulation
        res = rollout(node.pos, maze, selector)
        # backprop
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += res
            tmp = tmp.parent
        # stop if goal reached
        if maze.reached_target(node.pos):
            return root, i
    return root, None

def extract_path(node, maze):
    if maze.reached_target(node.pos):
        return node.build_path()
    for ch in node.children:
        sol = extract_path(ch, maze)
        if sol:
            return sol
    return []

# ==========================
# worker fns + parallel runner
# ==========================
def worker_classical(args):
    maze, its = args
    t0 = time.time()
    root,_ = mcts_stop(maze, its, classical_choice)
    return time.time()-t0

def worker_quantum(args):
    maze, its = args
    t0 = time.time()
    root,_ = mcts_stop(maze, its, quantum_choice)
    return time.time()-t0

def parallel_time(fn, maze, its, workers):
    tasks = [(maze, its)] * workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        times = list(ex.map(fn, tasks))
    return min(times)

# ==========================
# run 10 trials per maze & summary
# ==========================
def main():
    iterations = 200
    workers    = 8
    trials     = 20

    for m in preset_mazes:
        mz = Maze(m["grid"], m["start"], m["goal"])
        tc_list, tq_list = [], []
        wins_c, wins_q   = 0, 0

        for _ in range(trials):
            tc = parallel_time(worker_classical, mz, iterations, workers)
            tq = parallel_time(worker_quantum,   mz, iterations, workers)
            tc_list.append(tc)
            tq_list.append(tq)
            if   tc < tq: wins_c += 1
            elif tq < tc: wins_q += 1
            # ties don’t count

        avg_c = np.mean(tc_list)
        avg_q = np.mean(tq_list)

        print(f"\n=== Maze {m['name']} ===")
        print(f" Classical avg time: {avg_c:.4f}s, wins {wins_c}/{trials}")
        print(f" Quantum   avg time: {avg_q:.4f}s, wins {wins_q}/{trials}")

if __name__=="__main__":
    main()
