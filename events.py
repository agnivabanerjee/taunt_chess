
# events.py
from dataclasses import dataclass
from typing import Callable, List
import chess
from engine_utils import policy_moves

# --------- Helpers ---------
EXTENDED_CENTER = chess.SquareSet(
    chess.BB_C3 | chess.BB_D3 | chess.BB_E3 | chess.BB_F3 |
    chess.BB_C4 | chess.BB_D4 | chess.BB_E4 | chess.BB_F4 |
    chess.BB_C5 | chess.BB_D5 | chess.BB_E5 | chess.BB_F5 |
    chess.BB_C6 | chess.BB_D6 | chess.BB_E6 | chess.BB_F6
)

def piece_material_value(pt: int) -> int:
    # Only used if you later want material-delta events; keeping here if needed.
    return {chess.PAWN:100, chess.KNIGHT:300, chess.BISHOP:300, chess.ROOK:500, chess.QUEEN:900}.get(pt, 0)


def squares_around_king(board: chess.Board, color: bool) -> List[int]:
    ks = board.king(color)
    if ks is None:
        return []
    res = []
    kf, kr = chess.square_file(ks), chess.square_rank(ks)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                res.append(chess.square(f, r))
    return res

def is_passed_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    """Simple passed-pawn test (no opposing pawn in same/adjacent file ahead)."""
    piece = board.piece_at(sq)
    if not piece or piece.piece_type != chess.PAWN or piece.color != color:
        return False
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    step = 1 if color == chess.WHITE else -1
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    for df in (-1, 0, 1):
        f = file + df
        if 0 <= f <= 7:
            r = rank + step
            while 0 <= r <= 7:
                if (p := board.piece_at(chess.square(f, r))) and p.piece_type == chess.PAWN and p.color == enemy:
                    return False
                r += step
    return True

def knight_attacked_majors(board: chess.Board, sq: int, enemy: bool) -> int:
    """How many enemy majors (K, Q, R) are attacked by a knight on sq."""
    majors = {chess.KING, chess.QUEEN, chess.ROOK}
    cnt = 0
    for t in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[sq]):
        p = board.piece_at(t)
        if p and p.color == enemy and p.piece_type in majors:
            cnt += 1
    return cnt

def any_rook_on_7th(board: chess.Board, color: bool) -> bool:
    target_rank = 6 if color == chess.WHITE else 1  # 7th for white, 2nd for black
    for sq in board.pieces(chess.ROOK, color):
        if chess.square_rank(sq) == target_rank:
            return True
    return False

def king_ring_attackers(board: chess.Board, attacker: bool, defend_color: bool) -> int:
    ring = squares_around_king(board, defend_color)
    total = 0
    for sq in ring:
        total += len(board.attackers(attacker, sq))
    return total

# --- New, higher-frequency triggers ---

def give_check(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    # After our move, opponent to move is in check.
    moved_color = prev_b.turn
    return moved_color == side and new_b.is_check()

def capture_pawn(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    cap = prev_b.piece_at(mv.to_square)
    return cap is not None and cap.color != side and cap.piece_type == chess.PAWN and prev_b.turn == side

def attack_enemy_queen(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    enemy = not side
    qs = list(new_b.pieces(chess.QUEEN, enemy))
    if not qs:
        return False  # no queen to attack
    qsq = qs[0]
    return len(new_b.attackers(side, qsq)) > 0 and prev_b.turn == side

def center_entry(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    # Moved piece lands on extended center (16 squares).
    mover = prev_b.piece_at(mv.from_square)
    return mover is not None and mover.color == side and mv.to_square in EXTENDED_CENTER

def ring_poke(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    # Our attacks into enemy king ring strictly increase.
    enemy = not side
    before = king_ring_attackers(prev_b, side, enemy)
    after = king_ring_attackers(new_b, side, enemy)
    return prev_b.turn == side and after > before

def pawn_to_fifth(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    mover = prev_b.piece_at(mv.from_square)
    if not (mover and mover.color == side and mover.piece_type == chess.PAWN):
        return False
    r = chess.square_rank(mv.to_square)
    # White 5th rank = rank index 4; Black 5th rank = index 3
    return (side == chess.WHITE and r == 4) or (side == chess.BLACK and r == 3)

def rook_on_open_file(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    mover = prev_b.piece_at(mv.from_square)
    if not (mover and mover.color == side and mover.piece_type == chess.ROOK):
        return False
    file_idx = chess.square_file(mv.to_square)
    # Open file: no pawns of either color on that file anywhere.
    for sq in new_b.pieces(chess.PAWN, chess.WHITE) | new_b.pieces(chess.PAWN, chess.BLACK):
        if chess.square_file(sq) == file_idx:
            return False
    return True


# --------- Event definitions ---------

@dataclass
class EventDef:
    key: str
    label: str
    # (prev_board, move, new_board, side) -> bool  (side = prophesying side)
    trigger: Callable[[chess.Board, chess.Move, chess.Board, bool], bool]
    # describe odds text
    taunt: Callable[[float, int], str]

def taunt_text(template: str):
    def f(prob: float, k: int) -> str:
        pct = int(round(prob * 100))
        return template.format(p=pct, k=k)
    return f

def bishop_captures_rook(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    mover = prev_b.piece_at(mv.from_square)
    cap = prev_b.piece_at(mv.to_square)
    return (
        mover is not None and mover.color == side and mover.piece_type == chess.BISHOP and
        cap is not None and cap.color != side and cap.piece_type == chess.ROOK
    )

def knight_fork(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    mover = prev_b.piece_at(mv.from_square)
    if not mover or mover.color != side or mover.piece_type != chess.KNIGHT:
        return False
    enemy = not side
    return knight_attacked_majors(new_b, mv.to_square, enemy) >= 2

def passed_pawn_created(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    # Trigger if the moving pawn of 'side' becomes passed (or any pawn newly is).
    mover = prev_b.piece_at(mv.from_square)
    if mover and mover.color == side and mover.piece_type == chess.PAWN:
        return is_passed_pawn(new_b, mv.to_square, side)
    # also allow emergence by a capture elsewhere
    for sq in new_b.pieces(chess.PAWN, side):
        if is_passed_pawn(new_b, sq, side) and not is_passed_pawn(prev_b, sq, side):
            return True
    return False

def rook_on_seventh(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    mover = prev_b.piece_at(mv.from_square)
    if mover and mover.color == side and mover.piece_type == chess.ROOK:
        return any_rook_on_7th(new_b, side)
    return False

def king_ring_breach(prev_b: chess.Board, mv: chess.Move, new_b: chess.Board, side: bool) -> bool:
    # After the side's move, attackers into enemy king ring >= 3
    enemy = not side
    return king_ring_attackers(new_b, side, enemy) >= 3

EVENTS: List[EventDef] = [
    EventDef(
        key="bishop_x_rook",
        label="Bishop will take your rook",
        trigger=bishop_captures_rook,
        taunt=taunt_text("Prophecy: my bishop will harvest your rook within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="knight_fork",
        label="Knight fork on your royalty",
        trigger=knight_fork,
        taunt=taunt_text("Omen: my knight will fork your royalty within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="passed_pawn",
        label="A passed pawn will be born",
        trigger=passed_pawn_created,
        taunt=taunt_text("Foresight: a passed pawn will rise for me within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="rook_on_7th",
        label="Rook invasion on the 7th",
        trigger=rook_on_seventh,
        taunt=taunt_text("Sign: my rook will invade your 7th within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="king_ring_breach",
        label="Your king's ring will be breached",
        trigger=king_ring_breach,
        taunt=taunt_text("Warning: your king’s ring will be breached within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="give_check",
        label="Give check",
        trigger=give_check,
        taunt=taunt_text("Prophecy: I’ll check your king within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="capture_pawn",
        label="Capture a pawn",
        trigger=capture_pawn,
        taunt=taunt_text("Omen: I’ll snatch a pawn within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="attack_queen",
        label="Attack your queen",
        trigger=attack_enemy_queen,
        taunt=taunt_text("Warning: your queen will be under fire within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="center_entry",
        label="Invade the center",
        trigger=center_entry,
        taunt=taunt_text("Sign: I’ll plant a piece deep in the center within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="ring_poke",
        label="Poke the king’s ring",
        trigger=ring_poke,
        taunt=taunt_text("Foresight: pressure on your king will increase within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="pawn_to_5th",
        label="Pawn reaches the 5th",
        trigger=pawn_to_fifth,
        taunt=taunt_text("I foresee a pawn storm—one reaches the 5th within {k} plies. (≈{p}%)"),
    ),
    # Optional—keep if you want even more action
    EventDef(
        key="rook_open_file",
        label="Rook to an open file",
        trigger=rook_on_open_file,
        taunt=taunt_text("I’ll seize an open file with my rook within {k} plies. (≈{p}%)"),
    ),
]

EASY_EVENTS: List[EventDef] = [
    EventDef(
        key="give_check",
        label="Give check",
        trigger=give_check,
        taunt=taunt_text("Prophecy: I’ll check your king within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="capture_pawn",
        label="Capture a pawn",
        trigger=capture_pawn,
        taunt=taunt_text("Omen: I’ll snatch a pawn within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="attack_queen",
        label="Attack your queen",
        trigger=attack_enemy_queen,
        taunt=taunt_text("Warning: your queen will be under fire within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="center_entry",
        label="Invade the center",
        trigger=center_entry,
        taunt=taunt_text("Sign: I’ll plant a piece deep in the center within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="ring_poke",
        label="Poke the king’s ring",
        trigger=ring_poke,
        taunt=taunt_text("Foresight: pressure on your king will increase within {k} plies. (≈{p}%)"),
    ),
    EventDef(
        key="pawn_to_5th",
        label="Pawn reaches the 5th",
        trigger=pawn_to_fifth,
        taunt=taunt_text("I foresee a pawn storm—one reaches the 5th within {k} plies. (≈{p}%)"),
    ),
    # Optional—keep if you want even more action
    EventDef(
        key="rook_open_file",
        label="Rook to an open file",
        trigger=rook_on_open_file,
        taunt=taunt_text("I’ll seize an open file with my rook within {k} plies. (≈{p}%)"),
    ),
]


# --------- Probability via weighted tree ---------

def _estimate_total_nodes(plies: int, topk: int) -> int:
    # Sum_{i=1..plies} topk^i   (upper bound for move-considerations)
    if topk <= 1:
        return plies
    return topk * (topk**plies - 1) // (topk - 1)

def event_probability(engine, board: chess.Board, event: EventDef, side: bool,
                      plies: int = 6, topk: int = 4, depth: int = 12, temp: float = 1.0,
                      reporter=None, task_label: str | None = None) -> float:
    """
    Probability (under a depth-limited, weighted policy tree) that event triggers
    within ≤ plies moves. If 'reporter' and 'task_label' are provided, a progress
    bar shows approximate node exploration.
    """
    task = None
    if reporter and task_label:
        total = _estimate_total_nodes(plies, topk)
        task = reporter.start(f"{task_label}", total=total)

    def dfs(b: chess.Board, ply_left: int, mass: float) -> float:
        if ply_left == 0 or b.is_game_over():
            return 0.0
        prob = 0.0
        moves = policy_moves(engine, b, depth=depth, topk=topk, temp_pawns=temp)
        for mv, w, _cp in moves:
            if task: task.update(1)
            prev = b
            b2 = b.copy()
            b2.push(mv)
            moved_color = b.turn
            if moved_color == side and event.trigger(prev, mv, b2, side):
                prob += mass * w
            else:
                prob += dfs(b2, ply_left - 1, mass * w)
        return prob

    out = dfs(board, plies, 1.0)
    if task: task.close()
    return out