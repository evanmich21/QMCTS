import math
import random
import numpy as np
import time
import concurrent.futures
from math import ceil, log2
import pennylane as qml

# ==========================
# Mazes to test
# ==========================
preset_mazes = [
    {"name": "2×2",  "grid": [[0,1],[0,0]],                               "start": (0,0), "goal": (1,1)},
    {"name": "3×3",  "grid": [[0,0,0],[0,1,0],[0,0,0]],                     "start": (0,0), "goal": (2,2)},
    #{"name": "4×4",  "grid": [[0,0,0,0],[1,1,0,1],[0,0,0,0],[0,1,1,0]],     "start": (0,0), "goal": (3,3)},
    #{"name": "5×5",  "grid": [[0,0,0,0,0],[1,1,0,1,1],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0]], "start": (0,0), "goal": (4,4)},
    #{"name": "10×10","grid": [[0]*10 for _ in range(10)],                  "start": (0,0), "goal": (9,9)},
    #{"name": "20×20","grid": [[0]*20 for _ in range(20)],                  "start": (0,0), "goal": (19,19)},
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
        cand = [("up",(x,y-1)),("down",(x,y+1)),("left",(x-1,y)),("right",(x+1,y))]
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

# ==========================
# Precompile QNodes (1- and 2-wire)
# ==========================
DEV_Q1 = qml.device("default.qubit", wires=1)
@qml.qnode(DEV_Q1)
def circ_q1():
    qml.Hadamard(wires=0)
    return qml.probs(wires=[0])

DEV_Q2 = qml.device("default.qubit", wires=2)
@qml.qnode(DEV_Q2)
def circ_q2():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    return qml.probs(wires=[0,1])

_QNODES = {1: circ_q1, 2: circ_q2}
_GLOBAL_BATCH = {}
_GLOBAL_BATCH_SIZE = None

def quantum_choice(options):
    """Quantum-random choice with a dynamic batch size."""
    global _GLOBAL_BATCH_SIZE, _GLOBAL_BATCH
    n = len(options)
    if n == 1:
        return options[0]
    q = ceil(log2(n))
    if _GLOBAL_BATCH_SIZE is None:
        raise RuntimeError("Batch size not set before quantum_choice")
    if q not in _GLOBAL_BATCH or not _GLOBAL_BATCH[q]:
        probs = _QNODES[q]()  # returns length 2**q
        _GLOBAL_BATCH[q] = list(np.random.choice(
            len(probs),
            size=_GLOBAL_BATCH_SIZE,
            p=probs
        ))
    idx = _GLOBAL_BATCH[q].pop()
    return options[idx] if idx < n else quantum_choice(options)

# ==========================
# MCTS core
# ==========================
class MCTSNode:
    __slots__ = ("pos","parent","children","visits","wins","untried")
    def __init__(self, pos, parent=None):
        self.pos      = pos
        self.parent   = parent
        self.children = []
        self.visits   = 0
        self.wins     = 0
        self.untried  = None
    def fully_expanded(self, maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        return len(self.untried) == 0
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
            if ch.visits == 0:
                score = float("inf")
            else:
                score = (ch.wins/ch.visits) + c * math.sqrt(math.log(self.visits)/ch.visits)
            if score > best_score:
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
        # bias by Manhattan
        d0 = manhattan_distance(cur, maze.goal)
        better = [m for m in moves if manhattan_distance(m[1], maze.goal)<d0]
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
        # expand
        if not maze.reached_target(node.pos):
            node = node.expand(maze, selector)
        # simulate & backprop
        res = rollout(node.pos, maze, selector)
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += res
            tmp = tmp.parent
        # stop if goal
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
# worker fns + parallel runners
# ==========================
def worker_classical(args):
    maze, its = args
    t0 = time.time()
    mcts_stop(maze, its, classical_choice)
    return time.time() - t0

def worker_quantum(args):
    maze, its, batch_size = args
    global _GLOBAL_BATCH_SIZE, _GLOBAL_BATCH
    _GLOBAL_BATCH_SIZE = batch_size
    _GLOBAL_BATCH.clear()
    t0 = time.time()
    mcts_stop(maze, its, quantum_choice)
    return time.time() - t0

def parallel_time_classical(maze, iterations, workers):
    tasks = [(maze, iterations)] * workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        times = list(ex.map(worker_classical, tasks))
    return min(times)

def parallel_time_quantum(maze, iterations, batch_size, workers):
    tasks = [(maze, iterations, batch_size)] * workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        times = list(ex.map(worker_quantum, tasks))
    return min(times)

# ==========================
# Main: dynamic batch & iterations
# ==========================
def main():
    workers = 12
    trials  = 10

    for m in preset_mazes:
        mz = Maze(m["grid"], m["start"], m["goal"])
        area = mz.rows * mz.cols
        # dynamic batch & iterations
        batch_size = min(20000, max(250, area * 100))
        iterations = max(1000, area * 10)

        tc_list, tq_list = [], []
        wins_c, wins_q = 0, 0

        for _ in range(trials):
            tc = parallel_time_classical(mz, iterations, workers)
            tq = parallel_time_quantum(mz, iterations, batch_size, workers)
            tc_list.append(tc)
            tq_list.append(tq)
            if tc < tq:
                wins_c += 1
            elif tq < tc:
                wins_q += 1

        avg_c = np.mean(tc_list)
        avg_q = np.mean(tq_list)

        print(f"\n=== Maze {m['name']} (area={area}) ===")
        print(f" using batch_size={batch_size}, iterations={iterations}")
        print(f" Classical avg time: {avg_c*1000:.3f} ms, wins {wins_c}/{trials}")
        print(f" Quantum   avg time: {avg_q*1000:.3f} ms, wins {wins_q}/{trials}")

if __name__ == "__main__":
    main()
