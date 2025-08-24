# policy_tree.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set
import chess
from engine_utils import policy_moves, softmax
from progress_utils import ProgressReporter

# -------- Position keys (normalized) --------
def fen_key(board: chess.Board) -> Tuple[str, str, str, Optional[int]]:
    # Ignore halfmove/fullmove counters so equivalent positions collide in cache
    return (
        board.board_fen(),
        'w' if board.turn == chess.WHITE else 'b',
        board.castling_xfen() or '-',   # includes rights compactly
        board.ep_square,                # None or 0..63
    )

def board_from_key(key: Tuple[str, str, str, Optional[int]]) -> chess.Board:
    pieces, turn, castling, ep = key
    ep_field = chess.square_name(ep) if ep is not None else '-'
    fen = f"{pieces} {turn} {castling} {ep_field} 0 1"
    b = chess.Board(fen, chess960=False)
    return b

# -------- Tree structures --------
@dataclass
class Edge:
    uci: str
    weight: float
    cp: int
    child_idx: Optional[int]
    new_key: Tuple[str, str, str, Optional[int]]

@dataclass
class Node:
    key: Tuple[str, str, str, Optional[int]]
    edges: List[Edge]

@dataclass
class TreeParams:
    plies: int = 6
    depth_root: int = 12
    depth_inner: int = 10
    topk_root: int = 4
    topk_inner: int = 3
    temp_pawns: float = 1.0
    # Motif-aware widening caps
    widen_checks: int = 4
    widen_captures: int = 4
    widen_hooks: int = 2
    widen_lifts: int = 2
    widen_total_cap: int = 6
    quick_eval_depth: int = 6  # used to score widened moves not in MultiPV

class PolicyTree:
    def __init__(self, nodes: List[Node], root_idx: int, first_move_map: Dict[str, int]):
        self.nodes = nodes
        self.root_idx = root_idx
        self.first_map = first_move_map  # root move uci -> child idx

    def child_by_first(self, uci: str) -> Optional[int]:
        return self.first_map.get(uci)

# -------- Motif-aware widening (cheap heuristics) --------
def _generate_motif_moves(board: chess.Board, limit_caps: Dict[str, int]) -> List[chess.Move]:
    """Propose extra candidate moves likely to start flashy sequences."""
    side = board.turn
    enemy = not side
    legal = list(board.legal_moves)
    picks: List[chess.Move] = []
    seen: Set[str] = set()

    def add(m: chess.Move, tag: str):
        u = m.uci()
        if u in seen: return
        if limit_caps[tag] <= 0: return
        seen.add(u); picks.append(m); limit_caps[tag] -= 1

    # Checks
    for mv in legal:
        if board.gives_check(mv):
            add(mv, "checks")
            if limit_caps["checks"] == 0: break

    # Captures of majors / near-king captures
    majors = {chess.QUEEN, chess.ROOK}
    for mv in legal:
        cap = board.piece_at(mv.to_square)
        if cap and cap.color == enemy and cap.piece_type in majors:
            add(mv, "captures")

    # Hooks for h/g-file storms (push h- or g-pawns one step if legal)
    for file_name in ("h", "g"):
        for rank in (2, 3, 4, 5):  # try a few pushes
            try:
                from_sq = chess.parse_square(f"{file_name}{rank if side==chess.WHITE else 9-rank}")
            except ValueError:
                continue
            to_rank = rank + 1 if side == chess.WHITE else rank - 1
            if to_rank < 1 or to_rank > 8: continue
            to_sq = chess.parse_square(f"{file_name}{to_rank if side==chess.WHITE else 9-to_rank}")
            mv = chess.Move(from_sq, to_sq)
            if mv in legal:
                add(mv, "hooks")
                break

    # Rook lifts to 3rd / 6th rank (Ra3/Rh3 style)
    target_rank = 2 if side == chess.WHITE else 5  # zero-based ranks: 2 => 3rd, 5 => 6th
    for sq in board.pieces(chess.ROOK, side):
        f = chess.square_file(sq)
        for r in range(chess.square_rank(sq), target_rank - 1, -1) if side==chess.WHITE else range(chess.square_rank(sq), target_rank + 1):
            dst = chess.square(f, target_rank)
            mv = chess.Move(sq, dst)
            if mv in legal:
                add(mv, "lifts")
                break

    return picks

# -------- Tree builder with per-turn caches --------
class PolicyTreeBuilder:
    def __init__(self, engine, params: TreeParams, reporter: Optional[ProgressReporter] = None):
        self.engine = engine
        self.p = params
        self.prog = reporter or ProgressReporter(False)
        # cache: key -> list[(move, w, cp)]
        self._policy_cache: Dict[Tuple[Tuple[str,str,str,Optional[int]], int, int, float], List[Tuple[chess.Move,float,int]]] = {}

    def _policy_at(self, board: chess.Board, depth: int, topk: int) -> List[Tuple[chess.Move, float, int]]:
        key = (fen_key(board), depth, topk, self.p.temp_pawns)
        if key in self._policy_cache:
            return self._policy_cache[key]

        # Base engine picks
        engine_picks = policy_moves(self.engine, board, depth=depth, topk=topk, temp_pawns=self.p.temp_pawns)

        # Motif-aware widening
        limits = {"checks": self.p.widen_checks, "captures": self.p.widen_captures,
                  "hooks": self.p.widen_hooks, "lifts": self.p.widen_lifts}
        widen_moves = _generate_motif_moves(board, limits)

        # Merge & score widened moves that didn't appear
        have = {m.uci(): (m, w, cp) for (m, w, cp) in engine_picks}
        extra: List[Tuple[chess.Move,float,int]] = []
        for mv in widen_moves:
            if mv.uci() in have:  # already evaluated by engine
                continue
            # Quick evaluation (shallow) to assign a cp, then re-softmax
            b2 = board.copy(); b2.push(mv)
            # Depth shaved for speed
            info = self.engine.analyse(b2, chess.engine.Limit(depth=max(2, self.p.quick_eval_depth)))
            # negate because analysis was from opponent-to-move POV
            cp = -info["score"].pov(b2.turn).score(mate_score=100000) or 0
            extra.append((mv, 0.0, int(cp)))

        # merged = list(have.values()) + extra
        # # Recompute weights over all candidates using softmax(cp / (100*temp))
        # cps_scaled = [cp/(100.0*max(1e-9, self.p.temp_pawns)) for (_, _, cp) in merged]
        # weights = softmax(cps_scaled)
        # out = [(mv, w, cp) for (mv, (_, _, cp)), w in zip([m for m in [t[0] for t in merged]], zip(merged, weights))]
        # # Cap total candidates to avoid blow-up
        # out = sorted(out, key=lambda t: t[1], reverse=True)[:self.p.widen_total_cap + topk]
        # self._policy_cache[key] = out
        # return out

        # inside PolicyTreeBuilder._policy_at(...)
        merged = list(have.values()) + extra  # [(mv, w_old_or_0, cp)]
        cps_scaled = [cp / (100.0 * max(1e-9, self.p.temp_pawns)) for (_, _, cp) in merged]
        weights = softmax(cps_scaled)
        out = [ (mv, w, cp) for (mv, _, cp), w in zip(merged, weights) ]
        # cap and sort as before
        out = sorted(out, key=lambda t: t[1], reverse=True)[: self.p.widen_total_cap + topk]
        self._policy_cache[key] = out
        return out


    def build(self, root_board: chess.Board) -> PolicyTree:
        nodes: List[Node] = []
        first_map: Dict[str, int] = {}
        task = self.prog.start("Build shared tree", total=self.p.plies)

        def rec(board: chess.Board, depth_left: int) -> int:
            node_idx = len(nodes)
            nodes.append(Node(key=fen_key(board), edges=[]))
            if depth_left == 0 or board.is_game_over():
                return node_idx

            is_root = (depth_left == self.p.plies)
            depth = self.p.depth_root if is_root else self.p.depth_inner
            topk  = self.p.topk_root  if is_root else self.p.topk_inner

            for mv, w, cp in self._policy_at(board, depth=depth, topk=topk):
                board.push(mv)
                child_idx = rec(board, depth_left - 1)
                new_key = fen_key(board)
                board.pop()
                nodes[node_idx].edges.append(Edge(
                    uci=mv.uci(), weight=w, cp=cp, child_idx=child_idx, new_key=new_key
                ))
                if is_root and mv.uci() not in first_map:
                    first_map[mv.uci()] = child_idx

            if is_root:
                task.update(1)  # one tick for root layer
            return node_idx

        root_idx = rec(root_board.copy(), self.p.plies)
        task.close()
        return PolicyTree(nodes, root_idx, first_map)

# -------- Fast probability queries on a built tree --------
def probs_for_all_events(tree: PolicyTree, events, side: bool) -> Dict[str, float]:
    probs = {ev.key: 0.0 for ev in events}

    def dfs(node_idx: int, mass: float, prev_b: Optional[chess.Board]):
        node = tree.nodes[node_idx]
        prev_board = prev_b or board_from_key(node.key)
        for e in node.edges:
            m2 = mass * e.weight
            new_board = board_from_key(e.new_key)
            fired_any = False
            for ev in events:
                if ev.trigger(prev_board, chess.Move.from_uci(e.uci), new_board, side):
                    probs[ev.key] += m2
                    fired_any = True
            if not fired_any and e.child_idx is not None:
                dfs(e.child_idx, m2, new_board)

    dfs(tree.root_idx, 1.0, None)
    return probs

def prob_for_event_given_first(tree: PolicyTree, event, side: bool, first_uci: str) -> float:
    child_idx = tree.child_by_first(first_uci)
    if child_idx is None:
        return 0.0
    total = 0.0
    def dfs(node_idx: int, mass: float, prev_b: Optional[chess.Board]):
        nonlocal total
        node = tree.nodes[node_idx]
        prev_board = prev_b or board_from_key(node.key)
        for e in node.edges:
            m2 = mass * e.weight
            new_board = board_from_key(e.new_key)
            if event.trigger(prev_board, chess.Move.from_uci(e.uci), new_board, side):
                total += m2
            elif e.child_idx is not None:
                dfs(e.child_idx, m2, new_board)
    dfs(child_idx, 1.0, None)
    return total
