
# engine_utils.py
import os, shutil, math
import chess, chess.engine

INF_MATE_CP = 100_000  # mapped "infinity" for mate scores


def find_stockfish_path() -> str:
    p = os.environ.get("STOCKFISH_EXECUTABLE")
    if p and os.path.isfile(p):
        return p
    p = shutil.which("stockfish")
    if p:
        return p
    raise FileNotFoundError(
        "Stockfish not found. Put it on PATH as 'stockfish' or set STOCKFISH_EXECUTABLE."
    )


def make_engine(skill_level: int = 20, hash_mb: int = 256, threads: int = max(1, os.cpu_count() or 1)):
    """Create and configure a Stockfish engine process."""
    eng = chess.engine.SimpleEngine.popen_uci(find_stockfish_path())
    opts = {
        "Threads": threads,
        "Hash": hash_mb,
        # Many builds accept these:
        "Skill Level": max(0, min(20, skill_level)),
        # Some builds expose Elo limiting:
        "UCI_LimitStrength": False,
        "UCI_Elo": 2850,
    }
    for k, v in opts.items():
        try:
            eng.configure({k: v})
        except chess.engine.EngineError:
            pass
    return eng


def score_cp_from_pov(info_score: chess.engine.PovScore, turn: bool) -> int:
    """Centipawns from the POV of the side to move at the root."""
    s = info_score.pov(turn)
    if s.is_mate():
        m = s.mate()
        if m is None:
            return INF_MATE_CP
        sign = 1 if m > 0 else -1
        return sign * (INF_MATE_CP - 100 * abs(m))
    return int(s.score(mate_score=INF_MATE_CP) or 0)


def softmax(xs):
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def policy_moves(engine, board: chess.Board, depth: int = 12, topk: int = 4, temp_pawns: float = 1.0):
    """
    Get up to topk candidate root moves with weights ~ softmax(cp / (100*temp)).
    Returns list of (move, weight, cp_eval).
    """
    legal = list(board.legal_moves)
    if not legal:
        return []

    want = min(topk, len(legal))
    try:
        engine.configure({"MultiPV": want})
    except chess.engine.EngineError:
        pass

    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=want)
    if isinstance(infos, dict):
        infos = [infos]

    best_per_first = {}
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        mv0 = pv[0]
        cp = score_cp_from_pov(info["score"], board.turn)
        # keep best line per first-move
        key = mv0.uci()
        prev = best_per_first.get(key)
        if prev is None or cp > prev[1]:
            best_per_first[key] = (mv0, cp)

    # Fallback: if MultiPV was not helpful, probe each legal quickly (slower; rare)
    if not best_per_first:
        for mv in legal:
            b2 = board.copy()
            b2.push(mv)
            info = engine.analyse(b2, chess.engine.Limit(depth=max(2, depth - 2)))
            cp = -score_cp_from_pov(info["score"], b2.turn)
            best_per_first[mv.uci()] = (mv, cp)

    cps = [cp / (100.0 * max(1e-9, temp_pawns)) for (_, cp) in best_per_first.values()]
    probs = softmax(cps)
    out = [(mv, p, cp) for (mv, cp), p in zip(best_per_first.values(), probs)]
    # sort by weight
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def play_best(engine, board: chess.Board, depth: int = 14, movetime_ms: int | None = None) -> chess.Move:
    """Let engine choose its move."""
    limit = chess.engine.Limit(depth=depth) if movetime_ms is None else chess.engine.Limit(time=movetime_ms / 1000.0)
    res = engine.play(board, limit)
    return res.move
