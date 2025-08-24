# taunter.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import chess
# from events import EVENTS, event_probability, EventDef
from events import EVENTS, EASY_EVENTS, event_probability, EventDef
from engine_utils import policy_moves
from progress_utils import ProgressReporter
from policy_tree import PolicyTreeBuilder, TreeParams, probs_for_all_events, prob_for_event_given_first, fen_key, board_from_key

@dataclass
class Prophecy:
    event: EventDef
    side: bool
    issued_at_ply: int
    expires_at_ply: int
    horizon: int
    announced_text: str
    initial_prob: float
    status: str = "active"   # "active" | "fulfilled" | "averted"

@dataclass
class SteeringCfg:
    lam_cp: float = 25.0
    trust_region_drop_cp: int = 30
    candidate_topk: int = 4
    tree_topk: int = 3
    tree_depth: int = 12
    horizon_plies: int = 6
    temp_pawns: float = 1.0
    # Hybrid refine
    micro_deepen: bool = True
    refine_top_candidates: int = 2
    refine_extra_depth: int = 2  # add to display/inner depth when refining

@dataclass
class ReuseCfg:
    enabled: bool = True
    # params: TreeParams = TreeParams()
    params: TreeParams = field(default_factory=TreeParams)

class Taunter:
    def __init__(self, engine, side: bool,
                 prob_threshold: float = 0.35, horizon_plies: int = 6,
                 topk: int = 4, depth: int = 12, temp: float = 1.0,
                 cooldown_plies: int = 6, debug: bool = False,
                 reuse: Optional[ReuseCfg] = None, 
                 events: Optional[List[EventDef]] = None, avoid_retarget_plies: Optional[int] = None):
        self.engine = engine
        self.events: List[EventDef] = events if events is not None else EVENTS
        self.avoid_retarget_plies = avoid_retarget_plies if avoid_retarget_plies is not None else horizon_plies
        self.recently_completed: Dict[str, int] = {}   # event_key -> ban_until_ply
        self.side = side
        self.thresh = prob_threshold
        self.horizon = horizon_plies
        self.topk = topk
        self.depth = depth
        self.temp = temp
        self.cooldown = cooldown_plies
        self.last_taunt_ply = -999
        self.active: List[Prophecy] = []
        self.history: List[Prophecy] = []
        self.cached_target_event: Optional[EventDef] = None
        self.last_scan_ply: Optional[int] = None
        self.last_scan_fullmove: Optional[int] = None
        self.prog = ProgressReporter(enabled=debug)
        # reuse / shared tree
        self.reuse = reuse or ReuseCfg(enabled=True, params=TreeParams(plies=horizon_plies, depth_root=depth, depth_inner=max(2, depth-2), topk_root=topk, topk_inner=max(2, topk-1), temp_pawns=temp))
        self.shared_tree = None
        self.shared_tree_root_key = None
        self.shared_probs: dict[str, float] = {}

    # -------- schedule helpers --------
    def _is_banned(self, ev_key: str, now_ply: int) -> bool:
        until = self.recently_completed.get(ev_key)
        return until is not None and now_ply < until


    def current_ply(self, board: chess.Board) -> int:
        base = (board.fullmove_number - 1) * 2
        return base if board.turn == chess.WHITE else base + 1

    def is_scan_due(self, board: chess.Board, *, every_plies: int | None, every_fullmoves: int | None) -> bool:
        if every_plies is not None:
            now = self.current_ply(board)
            return self.last_scan_ply is None or now - (self.last_scan_ply or now) >= every_plies
        if every_fullmoves is not None:
            now = board.fullmove_number
            return self.last_scan_fullmove is None or now - (self.last_scan_fullmove or now) >= every_fullmoves
        return True

    # -------- scanning & taunting (reuse-aware) --------
    def _build_shared_tree_and_probs(self, board: chess.Board):
        if not self.reuse.enabled:
            self.shared_tree = None
            self.shared_probs = {}
            return
        builder = PolicyTreeBuilder(self.engine, self.reuse.params, reporter=self.prog)
        self.shared_tree = builder.build(board)
        self.shared_tree_root_key = fen_key(board)
        self.shared_probs = probs_for_all_events(self.shared_tree, self.events, self.side)

    # def scheduled_scan_and_maybe_taunt(self, board: chess.Board,
    #                                    *, every_plies: int | None, every_fullmoves: int | None):
    #     if not self.is_scan_due(board, every_plies=every_plies, every_fullmoves=every_fullmoves):
    #         return
    #     # one shared build + sweep
    #     self._build_shared_tree_and_probs(board)
    #     self.last_scan_ply = self.current_ply(board)
    #     self.last_scan_fullmove = board.fullmove_number
    #     # choose best
    #     picked = None
    #     for ev in self.events:
    #         p = self.shared_probs.get(ev.key, 0.0)
    #         if p >= self.thresh and (picked is None or p > picked[1]):
    #             picked = (ev, p)
    #     self.cached_target_event = picked[0] if picked else None
    #     # cooldown + taunt
    #     if picked:
    #         ev, p = picked
    #         now = self.current_ply(board)
    #         if now >= self.last_taunt_ply + self.cooldown:
    #             text = ev.taunt(p, self.horizon)
    #             pr = Prophecy(event=ev, side=self.side, issued_at_ply=now,
    #                           expires_at_ply=now + self.horizon, horizon=self.horizon,
    #                           announced_text=text, initial_prob=p)
    #             self.active.append(pr)
    #             self.last_taunt_ply = now
    #             print(f"\n🗣️  {text}")
    # taunter.py — in Taunter.scheduled_scan_and_maybe_taunt(...)
    def scheduled_scan_and_maybe_taunt(self, board, *, every_plies, every_fullmoves):
        if not self.is_scan_due(board, every_plies=every_plies, every_fullmoves=every_fullmoves):
            return

        picked = None
        if self.reuse.enabled:
            # build shared tree + sweep (hybrid path)
            self._build_shared_tree_and_probs(board)
            self.shared_tree_root_key = fen_key(board)
            # choose best from shared probs
            for ev in self.events:
                p = self.shared_probs.get(ev.key, 0.0)
                if p >= self.thresh and (picked is None or p > picked[1]):
                    picked = (ev, p)
        else:
            # ❗ legacy path: scan each event with event_probability
            # best = None
            # for ev in self.events:
            #     p = event_probability(self.engine, board, ev, self.side,
            #                         plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp)
            #     if p >= self.thresh and (best is None or p > best[1]):
            #         best = (ev, p)
            # picked = best
            # ❗ legacy path: scan each event with event_probability + progress bars
            scan_task = self.prog.start("Prophecy scan", total=len(self.events))
            best = None
            for ev in self.events:
                p = event_probability(
                    self.engine, board, ev, self.side,
                    plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp,
                    reporter=self.prog, task_label=f"Odds: {ev.key}"  # ← show per-event bar
                )
                if p >= self.thresh and (best is None or p > best[1]):
                    best = (ev, p)
                scan_task.update(1)
            scan_task.close()
            picked = best


        # bookkeeping
        self.last_scan_ply = self.current_ply(board)
        self.last_scan_fullmove = board.fullmove_number
        self.cached_target_event = picked[0] if picked else None

        # speak if above threshold & not on cooldown
        if picked:
            ev, p = picked
            now = self.current_ply(board)
            if now >= self.last_taunt_ply + self.cooldown:
                text = ev.taunt(p, self.horizon)
                self.active.append(Prophecy(
                    event=ev, side=self.side, issued_at_ply=now,
                    expires_at_ply=now + self.horizon, horizon=self.horizon,
                    announced_text=text, initial_prob=p))
                self.last_taunt_ply = now
                print(f"\n🗣️  {text}")

    # -------- state updates --------
    def update_after_move(self, prev_board: chess.Board, move: chess.Move, new_board: chess.Board):
        now = self.current_ply(new_board)
        moved_color = prev_board.turn
        to_remove = []
        for i, pr in enumerate(self.active):
            if pr.status != "active":
                to_remove.append(i); continue
            if now > pr.expires_at_ply:
                pr.status = "averted"; print(f"✨ Prophecy averted: {pr.event.label}")
                self.history.append(pr); to_remove.append(i); continue
            if moved_color == pr.side and pr.event.trigger(prev_board, move, new_board, pr.side):
                pr.status = "fulfilled"; print(f"🔥 Prophecy fulfilled: {pr.event.label} (on move {new_board.fullmove_number})")
                self.history.append(pr); to_remove.append(i)
        for i in reversed(to_remove):
            self.active.pop(i)

    # -------- steering --------
    def choose_move_with_steering(self, board: chess.Board, base_depth: int, cfg: SteeringCfg,
                                  *, allow_scan: bool = True) -> chess.Move:
        assert board.turn == self.side
        # Target selection: prefer active → cached (from last scan)
        target_event: Optional[EventDef] = None
        for pr in self.active:
            if pr.status == "active" and self.current_ply(board) <= pr.expires_at_ply:
                target_event = pr.event; break
        if target_event is None and self.cached_target_event is not None:
            target_event = self.cached_target_event
        # Optional ad-hoc scan if allowed and nothing to steer toward
        if target_event is None and allow_scan:
            # lightweight: if reuse is on, build tree once; else do classic select
            if self.reuse.enabled:
                self._build_shared_tree_and_probs(board)
                # pick best event even if below threshold for steering
                best_ev = max(self.events, key=lambda ev: self.shared_probs.get(ev.key, 0.0))
                target_event = best_ev
            else:
                # best = None
                # for ev in self.events:
                #     p = event_probability(self.engine, board, ev, self.side,
                #                           plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp)
                #     if best is None or p > best[1]: best = (ev, p)
                # target_event = best[0] if best else None
                task = self.prog.start("Opportunistic scan", total=len(self.events))
                best = None
                for ev in self.events:
                    p = event_probability(
                        self.engine, board, ev, self.side,
                        plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp,
                        reporter=self.prog, task_label=f"Odds: {ev.key}"  # ← per-event bar
                    )
                    if best is None or p > best[1]:
                        best = (ev, p)
                    task.update(1)
                task.close()
                target_event = best[0] if best else None


        # Candidate moves from engine
        cand = policy_moves(self.engine, board, depth=base_depth, topk=cfg.candidate_topk, temp_pawns=1.0)
        if not cand:
            return next(iter(board.legal_moves))
        best_cp = max(cp for _, _, cp in cand)

        # Compute P(event | first move) using the shared tree when possible
        use_shared = (self.reuse.enabled and self.shared_tree is not None and self.shared_tree_root_key == fen_key(board))
        scored = []
        for mv, _w, cp in cand:
            p = 0.0
            if target_event is not None:
                if use_shared:
                    p = prob_for_event_given_first(self.shared_tree, target_event, self.side, mv.uci())
                else:
                    # classic fallback (slower)
                    b2 = board.copy(); b2.push(mv)
                    p = event_probability(self.engine, b2, target_event, self.side,
                                          plies=cfg.horizon_plies, topk=cfg.tree_topk,
                                          depth=max(2, cfg.tree_depth-2), temp=cfg.temp_pawns)
            shaped = cp + cfg.lam_cp * p
            scored.append((mv, cp, shaped, p))

        # Optional hybrid micro-deepen: refine top-N candidates if probabilities are close/ambiguous
        if cfg.micro_deepen and target_event is not None and not use_shared and allow_scan:
            scored.sort(key=lambda t: t[2], reverse=True)
            for i, (mv, cp, _, p0) in enumerate(scored[:cfg.refine_top_candidates]):
                b2 = board.copy(); b2.push(mv)
                p_ref = event_probability(self.engine, b2, target_event, self.side,
                                          plies=cfg.horizon_plies,
                                          topk=max(2, cfg.tree_topk),
                                          depth=max(2, cfg.tree_depth + cfg.refine_extra_depth),
                                          temp=cfg.temp_pawns)
                shaped = cp + cfg.lam_cp * p_ref
                scored[i] = (mv, cp, shaped, p_ref)

        mv_star, cp_star, shaped_star, p_star = max(scored, key=lambda t: t[2])
        if best_cp - cp_star > cfg.trust_region_drop_cp:
            return max(cand, key=lambda t: t[2])[0]
        if target_event is not None and p_star >= 0.05:
            print(f"   ↳ Steering toward: {target_event.label} (Δeval≈{int(best_cp-cp_star)}cp, p≈{int(p_star*100)}%)")
        return mv_star

    # -------- display --------
    def render_status(self, board: chess.Board, *, show_current_probs: bool = True,
                      disp_depth: Optional[int] = None, disp_topk: Optional[int] = None,
                      max_history: int = 6):
        now = self.current_ply(board)
        active = self.active
        recent = self.history[-max_history:]

        print("\n== Active prophecies ==" if active else "\n== Active prophecies == (none)")
        for pr in active:
            left = max(0, pr.expires_at_ply - now)
            if show_current_probs:
                # Prefer shared-tree quick odds when possible
                p = None
                if self.reuse.enabled and self.shared_tree is not None and self.shared_tree_root_key == fen_key(board):
                    p = self.shared_probs.get(pr.event.key, pr.initial_prob)
                else:
                    p = pr.initial_prob
            else:
                p = pr.initial_prob
            pct = int(round(p * 100))
            print(f" • {pr.event.label} — expires in {left} plies — odds≈{pct}%")

        print("\n== Recent prophecies ==" if recent else "\n== Recent prophecies == (none)")
        for pr in reversed(recent):
            print(f" • {pr.event.label} — {pr.status} — issued at ply {pr.issued_at_ply}, horizon {pr.horizon}")
    
    def render_event_probabilities(
        self,
        board: chess.Board,
        *,
        prefer_shared: bool = True,
        disp_depth: int = 8,
        disp_topk: int = 3,
        temp: float | None = None,
    ):
        """
        Print a small dashboard of per-event probabilities from the current position.
        - If a shared tree for this exact board exists, reuse it (fast).
        - Otherwise, do a shallow, display-only scan for each event (cheap-ish).
        """
        # Try to reuse the shared tree if it matches this board
        using_shared = (
            prefer_shared
            and self.reuse.enabled
            and self.shared_tree is not None
            and self.shared_tree_root_key == fen_key(board)
        )

        probs: dict[str, float] = {}
        if using_shared:
            # Reuse what we already computed this turn
            for ev in self.events:
                probs[ev.key] = self.shared_probs.get(ev.key, 0.0)
        else:
            # Lightweight, display-only odds (shallower than steering)
            for ev in self.events:
                p = event_probability(
                    self.engine,
                    board,
                    ev,
                    self.side,
                    plies=min(self.horizon, 6),
                    topk=max(2, disp_topk),
                    depth=max(2, disp_depth),
                    temp=temp or self.temp,
                )
                probs[ev.key] = p

        # Pretty print
        title = "== Event probabilities (shared tree) ==" if using_shared else "== Event probabilities (display scan) =="
        print("\n" + title)
        # 20-char bar (5% per block)
        def bar(pct: int) -> str:
            filled = min(20, pct // 5)
            return "█" * filled + "·" * (20 - filled)

        for ev in self.events:
            pct = int(round(100 * probs.get(ev.key, 0.0)))
            print(f" • {ev.label:<30} [{bar(pct)}] {pct:3d}%")
