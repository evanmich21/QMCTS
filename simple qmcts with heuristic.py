import math, random, time
import numpy as np
import pennylane as qml

# === Preset Mazes ===
preset_mazes = [
    {"name": "2×2", "grid": [[0,1],[0,0]], "start":(0,0), "goal":(1,1)},
    {"name": "3×3", "grid": [[0,0,0],[0,1,0],[0,0,0]], "start":(0,0), "goal":(2,2)},
    {"name": "4×4", "grid": [
        [0,0,0,0],
        [1,1,0,1],
        [0,0,0,0],
        [0,1,1,0]
    ], "start":(0,0), "goal":(3,3)},
    {"name": "5×5", "grid": [
        [0,0,0,0,0],
        [1,1,0,1,1],
        [0,0,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ], "start":(0,0), "goal":(4,4)},
]

# === Maze Class ===
class Maze:
    def __init__(self, grid, start, goal):
        self.grid = grid
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
    def reached(self,pos):
        return pos==self.goal

# === Classical selector ===
def classical_choice(options):
    return random.choice(options)

# === Quantum selector with batching ===
_qnodes = {}
_batch  = {}
_BATCH_SIZE = 10000

def quantum_choice(options):
    n = len(options)
    if n==1:
        return options[0]
    q = math.ceil(math.log2(n))
    if q not in _qnodes:
        dev = qml.device("default.qubit", wires=q)
        @qml.qnode(dev)
        def circ():
            for i in range(q):
                qml.Hadamard(wires=i)
            return qml.probs(wires=range(q))
        _qnodes[q] = circ
    if q not in _batch or not _batch[q]:
        probs = _qnodes[q]()[:2**q]
        idxs  = np.arange(2**q)
        _batch[q] = list(np.random.choice(idxs, size=_BATCH_SIZE, p=probs))
    idx = _batch[q].pop()
    if idx < n:
        return options[idx]
    else:
        return quantum_choice(options)

# === UCT Node ===
C=1.4
class Node:
    __slots__ = ("pos","parent","children","visits","wins","untried")
    def __init__(self,pos,parent=None):
        self.pos      = pos
        self.parent   = parent
        self.children = []      # list of (move,Node)
        self.visits   = 0
        self.wins     = 0
        self.untried  = None
    def fully_expanded(self,maze):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        return len(self.untried)==0

    # classical expand
    def expand_classical(self,maze,selector):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = Node(mv[1], self)
        self.children.append((mv[0],child))
        return child

    # quantum expand with heuristic prior
    def expand_quantum(self,maze,selector):
        if self.untried is None:
            self.untried = maze.get_moves(self.pos).copy()
        mv = selector(self.untried)
        self.untried.remove(mv)
        child = Node(mv[1], self)
        # heuristic prior: closer to goal → small virtual win
        maxd = maze.rows + maze.cols - 2
        d    = abs(child.pos[0]-maze.goal[0]) + abs(child.pos[1]-maze.goal[1])
        prior = (maxd - d)/maxd
        child.visits = 1
        child.wins   = prior
        self.children.append((mv[0],child))
        return child

    def best_child(self):
        best_score=-1e9
        best=None
        for mv,ch in self.children:
            if ch.visits==0:
                score=1e9
            else:
                exploit=ch.wins/ch.visits
                explore=C*math.sqrt(math.log(self.visits)/ch.visits)
                score=exploit+explore
            if score>best_score:
                best_score, best = score, ch
        return best

# === Rollout ===  
def rollout(start,maze,selector,max_steps=50):
    cur=start
    for _ in range(max_steps):
        if maze.reached(cur):
            return 1
        moves=maze.get_moves(cur)
        if not moves:
            break
        cur=selector(moves)[1]
    return 0

# === MCTS stopping on goal classical ===
def mcts_stop_classical(maze,iterations):
    root=Node(maze.start)
    for it in range(1,iterations+1):
        node=root
        # select
        while node.fully_expanded(maze) and node.children and not maze.reached(node.pos):
            node=node.best_child()
        # expand
        if not maze.reached(node.pos):
            node=node.expand_classical(maze, classical_choice)
        # simulate
        res=rollout(node.pos,maze, classical_choice)
        # backprop
        tmp=node
        while tmp:
            tmp.visits+=1
            tmp.wins  +=res
            tmp=tmp.parent
        if maze.reached(node.pos):
            return root, it
    return root, None

# === MCTS stopping on goal quantum ===
def mcts_stop_quantum(maze,iterations):
    root=Node(maze.start)
    for it in range(1,iterations+1):
        node=root
        while node.fully_expanded(maze) and node.children and not maze.reached(node.pos):
            node=node.best_child()
        if not maze.reached(node.pos):
            node=node.expand_quantum(maze, quantum_choice)
        res=rollout(node.pos,maze, quantum_choice)
        tmp=node
        while tmp:
            tmp.visits+=1
            tmp.wins  +=res
            tmp=tmp.parent
        if maze.reached(node.pos):
            return root, it
    return root, None

def extract_path(node,maze):
    if maze.reached(node.pos):
        path=[]
        cur=node
        while cur:
            path.append(cur.pos)
            cur=cur.parent
        return path[::-1]
    for _,ch in node.children:
        sol=extract_path(ch,maze)
        if sol: return sol
    return None

# === compare ===
def run_one(m):
    print(f"\n--- Maze {m['name']} ---")
    mz = Maze(m["grid"], m["start"], m["goal"])
    its = 2000

    t0=time.time()
    root_c,ic = mcts_stop_classical(mz,its)
    tc=time.time()-t0
    sol_c=extract_path(root_c,mz)
    print(f"Classical   time={tc:.6f}s  iters={ic}  path={sol_c}")

    t1=time.time()
    root_q,iq = mcts_stop_quantum(mz,its)
    tq=time.time()-t1
    sol_q=extract_path(root_q,mz)
    print(f"Quantum     time={tq:.6f}s  iters={iq}  path={sol_q}")

def main():
    for m in preset_mazes:
        run_one(m)

if __name__=="__main__":
    main()
