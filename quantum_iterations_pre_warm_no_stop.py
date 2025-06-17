import math
import random
import time
from math import ceil, log2
from collections import deque

import numpy as np
import pennylane as qml
import concurrent.futures
import matplotlib.pyplot as plt

# ==========================
# Maze generation + solver
# ==========================
def is_solvable(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    q = deque([start])
    seen = {start}
    while q:
        x,y = q.popleft()
        if (x,y) == goal:
            return True
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx, y+dy
            if 0 <= nx < cols and 0 <= ny < rows \
               and grid[ny][nx] == 0 and (nx,ny) not in seen:
                seen.add((nx,ny))
                q.append((nx,ny))
    return False

def generate_solvable_maze(rows, cols, start, goal, wall_prob=0.3):
    sx,sy = start
    gx,gy = goal
    while True:
        grid = [
            [1 if random.random() < wall_prob else 0
             for _ in range(cols)]
            for _ in range(rows)
        ]
        grid[sy][sx] = 0
        grid[gy][gx] = 0
        if is_solvable(grid, start, goal):
            return grid

def print_maze(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    sx,sy = start
    gx,gy = goal
    for y in range(rows):
        line = []
        for x in range(cols):
            if (x,y)==(sx,sy):
                line.append("S")
            elif (x,y)==(gx,gy):
                line.append("G")
            elif grid[y][x]==1:
                line.append("#")
            else:
                line.append(".")
        print("".join(line))
    print()

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
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] == 0
    def get_moves(self, pos):
        x,y = pos
        cand = [("up",(x,y-1)),("down",(x,y+1)),
                ("left",(x-1,y)),("right",(x+1,y))]
        return [(d,p) for d,p in cand if self.is_valid(p)]
    def reached_target(self, pos):
        return pos == self.goal

# ==========================
# Helpers
# ==========================
def manhattan_distance(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def classical_choice(options):
    return random.choice(options)

# ==========================
# Quantum RNG + batching
# ==========================
_GLOBAL_QNODES = {}
_GLOBAL_BATCH  = {}
_BATCH_SIZE    = None

def init_worker(batch_size):
    global _BATCH_SIZE, _GLOBAL_QNODES, _GLOBAL_BATCH
    _BATCH_SIZE = batch_size
    _GLOBAL_BATCH = {}
    for q in (1,2):
        dev = qml.device("default.qubit", wires=q)
        @qml.qnode(dev)
        def circ(wires=range(q)):
            for i in wires:
                qml.Hadamard(wires=i)
            return qml.probs(wires=wires)
        _GLOBAL_QNODES[q] = circ
        probs = circ()
        _GLOBAL_BATCH[q] = list(np.random.choice(2**q, size=_BATCH_SIZE, p=probs))

def quantum_choice(options):
    n = len(options)
    if n == 1:
        return options[0]
    q = ceil(log2(n))
    if q not in _GLOBAL_BATCH or not _GLOBAL_BATCH[q]:
        probs = _GLOBAL_QNODES[q]()
        _GLOBAL_BATCH[q] = list(np.random.choice(2**q, size=_BATCH_SIZE, p=probs))
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
                score = (ch.wins/ch.visits) + c*math.sqrt(math.log(self.visits)/ch.visits)
            if score > best_score:
                best_score, best = score, ch
        return best
    def build_path(self):
        path, cur = [], self
        while cur:
            path.append(cur.pos)
            cur = cur.parent
        return path[::-1]

def rollout(start, maze, selector, max_steps=50):
    cur = start
    for _ in range(max_steps):
        if maze.reached_target(cur):
            return 1
        moves = maze.get_moves(cur)
        if not moves:
            break
        d0 = manhattan_distance(cur, maze.goal)
        better = [m for m in moves if manhattan_distance(m[1], maze.goal)<d0]
        if better:
            moves = better
        cur = selector(moves)[1]
    return 0

def mcts_stop(maze, iterations, selector):
    root = MCTSNode(maze.start)
    # run all iterations without early exit
    for _ in range(iterations):
        node = root
        while node.fully_expanded(maze) and node.children and not maze.reached_target(node.pos):
            node = node.best_child()
        if not maze.reached_target(node.pos):
            node = node.expand(maze, selector)
        res = rollout(node.pos, maze, selector)
        tmp = node
        while tmp:
            tmp.visits += 1
            tmp.wins   += res
            tmp = tmp.parent
    return root, None

# ==========================
# Worker functions
# ==========================
def worker_classical(args):
    maze, its = args
    t0 = time.time()
    mcts_stop(maze, its, classical_choice)
    return time.time() - t0

def worker_quantum(args):
    maze, its = args
    t0 = time.time()
    mcts_stop(maze, its, quantum_choice)
    return time.time() - t0

# ==========================
# Parallel timing fns
# ==========================
def parallel_time_classical(maze, iter_count, workers):
    tasks = [(maze, iter_count)] * workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        return min(ex.map(worker_classical, tasks))

def parallel_time_quantum(maze, iter_count, batch_size, workers):
    tasks = [(maze, iter_count)] * workers
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(batch_size,)
        ) as ex:
        return min(ex.map(worker_quantum, tasks))

# ==========================
# Main: generate + test + final summary + bar chart
# ==========================
def main():
    workers   = 8
    trials    = 100
    wall_prob = 0.3

    sizes       = []
    avg_c_times = []
    avg_q_times = []
    wins_c_list = []
    wins_q_list = []

    for (rows,cols) in ((2,2),(3,3),(4,4),(5,5),(10,10),(20,20)):
        area = rows * cols
        start = (0,0)
        goal  = (cols-1, rows-1)

        batch_size = min(20000, max(250, area*80))
        iterations = min(1000, area*8)

        tc_list, tq_list = [], []
        wins_c, wins_q   = 0, 0

        print(f"\n=== Testing {rows}×{cols} mazes ===")
        for t in range(1, trials+1):
            grid = generate_solvable_maze(rows, cols, start, goal, wall_prob)
            print(f"\nMaze #{t}:")
            print_maze(grid, start, goal)

            mz = Maze(grid, start, goal)
            tc = parallel_time_classical(mz, iterations, workers)
            tq = parallel_time_quantum(mz, iterations, batch_size, workers)

            tc_list.append(tc)
            tq_list.append(tq)
            if   tc < tq: wins_c += 1
            elif tq < tc: wins_q += 1

        avg_c = np.mean(tc_list)*1e3
        avg_q = np.mean(tq_list)*1e3

        sizes.append(f"{rows}×{cols}")
        avg_c_times.append(avg_c)
        avg_q_times.append(avg_q)
        wins_c_list.append(wins_c)
        wins_q_list.append(wins_q)

        print(f"\n--- {rows}×{cols} summary ---")
        print(f" batch_size={batch_size}, iterations={iterations}")
        print(f" Classical avg: {avg_c:.3f} ms, wins {wins_c}/{trials}")
        print(f" Quantum   avg: {avg_q:.3f} ms, wins {wins_q}/{trials}")

    # Final consolidated summary
    print("\n\n=== FINAL SUMMARY ===")
    print(f"{'Size':>6}  {'C_ms':>8}  {'Q_ms':>8}  {'C_wins':>6}  {'Q_wins':>6}")
    for sz, c, q, wc, wq in zip(sizes, avg_c_times, avg_q_times, wins_c_list, wins_q_list):
        print(f"{sz:>6}  {c:8.3f}  {q:8.3f}  {wc:6d}  {wq:6d}")

    # final grouped bar chart
    x = np.arange(len(sizes))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, avg_c_times, width, label="Classical")
    plt.bar(x + width/2, avg_q_times, width, label="Quantum")
    plt.xticks(x, sizes)
    plt.xlabel("Maze size")
    plt.ylabel("Average Time (ms)")
    plt.title("Classical vs Quantum Average Solve Times")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
