# taunter.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import chess

# Event system
from events import EventDef, event_probability, EVENTS  # pass a custom list at init if desired

# Engine helpers
from engine_utils import policy_moves

# Progress (debug-friendly bars/markers)
from progress_utils import ProgressReporter

# Shared reusable policy tree (hybrid path)
from policy_tree import (
    PolicyTreeBuilder, TreeParams, probs_for_all_events, prob_for_event_given_first, fen_key
)

# =========================
# Data classes / configs
# =========================

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
    lam_cp: float = 25.0                # bonus (cp) for +1.0 event probability
    trust_region_drop_cp: int = 30      # max eval loss vs best we accept
    candidate_topk: int = 4
    tree_topk: int = 3                  # legacy per-candidate branching
    tree_depth: int = 12                # legacy per-candidate depth
    horizon_plies: int = 6
    temp_pawns: float = 1.0
    # Hybrid refine (legacy path only; shared-tree path generally doesn’t need it)
    micro_deepen: bool = False
    refine_top_candidates: int = 2
    refine_extra_depth: int = 2

@dataclass
class ReuseCfg:
    enabled: bool = True
    params: TreeParams = field(default_factory=TreeParams)  # avoid mutable default error

# =========================
# Taunter
# =========================

class Taunter:
    def __init__(
        self,
        engine,
        side: bool,
        prob_threshold: float = 0.35,
        horizon_plies: int = 6,
        topk: int = 4,
        depth: int = 12,
        temp: float = 1.0,
        cooldown_plies: int = 6,
        debug: bool = False,
        reuse: Optional[ReuseCfg] = None,
        events: Optional[List[EventDef]] = None,
        avoid_retarget_plies: Optional[int] = None,
    ):
        self.engine = engine
        self.side = side
        self.thresh = prob_threshold
        self.horizon = horizon_plies
        self.topk = topk
        self.depth = depth
        self.temp = temp
        self.cooldown = cooldown_plies
        self.last_taunt_ply = -999

        # Prophecies and history
        self.active: List[Prophecy] = []
        self.history: List[Prophecy] = []

        # Scheduling & caching
        self.cached_target_event: Optional[EventDef] = None
        self.last_scan_ply: Optional[int] = None
        self.last_scan_fullmove: Optional[int] = None

        # Events to consider
        self.events: List[EventDef] = events if events is not None else EVENTS

        # Progress reporter (enabled with --debug)
        self.prog = ProgressReporter(enabled=debug)

        # Reuse / shared-tree config
        if reuse is None:
            reuse = ReuseCfg(enabled=True, params=TreeParams(
                plies=horizon_plies,
                depth_root=depth,
                depth_inner=max(2, depth - 2),
                topk_root=topk,
                topk_inner=max(2, topk - 1),
                temp_pawns=temp
            ))
        self.reuse = reuse
        self.shared_tree = None
        self.shared_tree_root_key = None
        self.shared_probs: Dict[str, float] = {}

        # Retarget ban (avoid picking the just-fulfilled event again immediately)
        self.avoid_retarget_plies = avoid_retarget_plies if avoid_retarget_plies is not None else horizon_plies
        self.recently_completed: Dict[str, int] = {}   # event_key -> ban_until_ply

    # ------------- utils -------------

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

    def _is_banned(self, ev_key: str, now_ply: int) -> bool:
        until = self.recently_completed.get(ev_key)
        return until is not None and now_ply < until

    # ------------- shared-tree build & sweep -------------

    def _build_shared_tree_and_probs(self, board: chess.Board):
        """Build a shared policy tree once and compute probabilities for all events."""
        if not self.reuse.enabled:
            self.shared_tree = None
            self.shared_probs = {}
            self.shared_tree_root_key = None
            return
        builder = PolicyTreeBuilder(self.engine, self.reuse.params, reporter=self.prog)
        self.shared_tree = builder.build(board)
        self.shared_tree_root_key = fen_key(board)
        self.shared_probs = probs_for_all_events(self.shared_tree, self.events, self.side)

    # ------------- scheduled scan & taunt -------------

    def scheduled_scan_and_maybe_taunt(self, board: chess.Board, *, every_plies: int | None, every_fullmoves: int | None):
        """Run a scan only when due; in reuse mode build/sweep once, else legacy per-event with progress bars."""
        if not self.is_scan_due(board, every_plies=every_plies, every_fullmoves=every_fullmoves):
            return

        now = self.current_ply(board)
        picked: Optional[Tuple[EventDef, float]] = None

        if self.reuse.enabled:
            self._build_shared_tree_and_probs(board)
            # choose best event above threshold, skipping banned
            for ev in self.events:
                if self._is_banned(ev.key, now):
                    continue
                p = self.shared_probs.get(ev.key, 0.0)
                if p >= self.thresh and (picked is None or p > picked[1]):
                    picked = (ev, p)
        else:
            # Legacy per-event scan with debug bars
            scan_task = self.prog.start("Prophecy scan", total=len(self.events))
            best = None
            for ev in self.events:
                if self._is_banned(ev.key, now):
                    scan_task.update(1)
                    continue
                p = event_probability(
                    self.engine, board, ev, self.side,
                    plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp,
                    reporter=self.prog, task_label=f"Odds: {ev.key}"
                )
                if p >= self.thresh and (best is None or p > best[1]):
                    best = (ev, p)
                scan_task.update(1)
            scan_task.close()
            picked = best

        # bookkeeping
        self.last_scan_ply = now
        self.last_scan_fullmove = board.fullmove_number
        self.cached_target_event = picked[0] if picked else None

        # Announce (cooldown + threshold)
        if picked:
            ev, p = picked
            if now >= self.last_taunt_ply + self.cooldown:
                text = ev.taunt(p, self.horizon)
                pr = Prophecy(
                    event=ev, side=self.side, issued_at_ply=now,
                    expires_at_ply=now + self.horizon, horizon=self.horizon,
                    announced_text=text, initial_prob=p
                )
                self.active.append(pr)
                self.last_taunt_ply = now
                print(f"\n🗣️  {text}")

    # ------------- game updates (fulfill / avert) -------------

    def update_after_move(self, prev_board: chess.Board, move: chess.Move, new_board: chess.Board):
        now = self.current_ply(new_board)
        moved_color = prev_board.turn
        to_remove = []

        for i, pr in enumerate(self.active):
            if pr.status != "active":
                to_remove.append(i)
                continue

            # Averted?
            if now > pr.expires_at_ply:
                pr.status = "averted"
                print(f"✨ Prophecy averted: {pr.event.label}")
                self.history.append(pr)
                # Optionally ban aversions for shorter time to enforce variety
                self.recently_completed[pr.event.key] = now + max(1, self.avoid_retarget_plies // 2)
                if self.cached_target_event and self.cached_target_event.key == pr.event.key:
                    self.cached_target_event = None
                to_remove.append(i)
                continue

            # Fulfilled? (only when prophet side just moved)
            if moved_color == pr.side and pr.event.trigger(prev_board, move, new_board, pr.side):
                pr.status = "fulfilled"
                print(f"🔥 Prophecy fulfilled: {pr.event.label} (on move {new_board.fullmove_number})")
                self.history.append(pr)
                # Ban this event for a while; clear cache if it was the target
                self.recently_completed[pr.event.key] = now + max(1, self.avoid_retarget_plies)
                if self.cached_target_event and self.cached_target_event.key == pr.event.key:
                    self.cached_target_event = None
                to_remove.append(i)

        for i in reversed(to_remove):
            self.active.pop(i)

    # ------------- move choice with steering -------------

    def choose_move_with_steering(self, board: chess.Board, base_depth: int, cfg: SteeringCfg, *, allow_scan: bool = True) -> chess.Move:
        """Pick a move; steer toward an event (active > cached > best-available), respecting trust region."""
        assert board.turn == self.side
        now = self.current_ply(board)

        # 1) Select target event without unnecessary rescans
        target_event: Optional[EventDef] = None

        # Prefer an active prophecy (not banned and not expired)
        for pr in self.active:
            if pr.status == "active" and now <= pr.expires_at_ply and not self._is_banned(pr.event.key, now):
                target_event = pr.event
                break

        # Else use cached target from last scheduled scan (if not banned)
        if target_event is None and self.cached_target_event is not None:
            if not self._is_banned(self.cached_target_event.key, now):
                target_event = self.cached_target_event
            else:
                self.cached_target_event = None  # drop banned cache

        # Else opportunistic pick (reuse shared probs if available), only if scanning is allowed
        if target_event is None and allow_scan:
            if self.reuse.enabled:
                # (Re)build shared tree for current node, then choose best non-banned
                self._build_shared_tree_and_probs(board)
                viable = [(ev, self.shared_probs.get(ev.key, 0.0))
                          for ev in self.events if not self._is_banned(ev.key, now)]
                if viable:
                    target_event = max(viable, key=lambda t: t[1])[0]
            else:
                # Legacy: quick per-event scan with bars
                task = self.prog.start("Opportunistic scan", total=len(self.events))
                best = None
                for ev in self.events:
                    if self._is_banned(ev.key, now):
                        task.update(1)
                        continue
                    p = event_probability(
                        self.engine, board, ev, self.side,
                        plies=self.horizon, topk=self.topk, depth=self.depth, temp=self.temp,
                        reporter=self.prog, task_label=f"Odds: {ev.key}"
                    )
                    if best is None or p > best[1]:
                        best = (ev, p)
                    task.update(1)
                task.close()
                target_event = best[0] if best else None

        # As last resort: if reuse tree matches and no scan allowed, pick highest shared prob (excluding banned)
        if target_event is None and self.reuse.enabled and self.shared_tree is not None and self.shared_tree_root_key == fen_key(board):
            viable = [(ev, self.shared_probs.get(ev.key, 0.0))
                      for ev in self.events if not self._is_banned(ev.key, now)]
            if viable:
                target_event = max(viable, key=lambda t: t[1])[0]

        # 2) Score candidate root moves (eval + steering bonus)
        cand = policy_moves(self.engine, board, depth=base_depth, topk=cfg.candidate_topk, temp_pawns=1.0)
        if not cand:
            return next(iter(board.legal_moves))
        best_cp = max(cp for _, _, cp in cand)

        use_shared = self.reuse.enabled and self.shared_tree is not None and self.shared_tree_root_key == fen_key(board)

        scored: List[Tuple[chess.Move, int, float, float]] = []
        for mv, _w, cp in cand:
            p = 0.0
            if target_event is not None:
                if use_shared:
                    # Cheap: P(event | first move) via shared tree
                    p = prob_for_event_given_first(self.shared_tree, target_event, self.side, mv.uci())
                else:
                    # Legacy fallback: fresh probability eval after the move
                    b2 = board.copy(); b2.push(mv)
                    p = event_probability(
                        self.engine, b2, target_event, self.side,
                        plies=cfg.horizon_plies, topk=cfg.tree_topk,
                        depth=max(2, cfg.tree_depth - 2), temp=cfg.temp_pawns
                    )
            shaped = cp + cfg.lam_cp * p
            scored.append((mv, cp, shaped, p))

        # Optional micro-deepen (legacy path only, when not using shared-tree odds)
        if cfg.micro_deepen and target_event is not None and not use_shared and allow_scan:
            scored.sort(key=lambda t: t[2], reverse=True)
            for i, (mv, cp, _, _) in enumerate(scored[:cfg.refine_top_candidates]):
                b2 = board.copy(); b2.push(mv)
                p_ref = event_probability(
                    self.engine, b2, target_event, self.side,
                    plies=cfg.horizon_plies,
                    topk=max(2, cfg.tree_topk),
                    depth=max(2, cfg.tree_depth + cfg.refine_extra_depth),
                    temp=cfg.temp_pawns
                )
                shaped = cp + cfg.lam_cp * p_ref
                scored[i] = (mv, cp, shaped, p_ref)

        # Pick shaped best, but respect trust region
        mv_star, cp_star, shaped_star, p_star = max(scored, key=lambda t: t[2])
        if best_cp - cp_star > cfg.trust_region_drop_cp:
            return max(cand, key=lambda t: t[2])[0]

        if target_event is not None and p_star >= 0.05:
            print(f"   ↳ Steering toward: {target_event.label} (Δeval≈{int(best_cp - cp_star)}cp, p≈{int(p_star * 100)}%)")
        return mv_star

    # ------------- displays -------------

    def render_status(
        self,
        board: chess.Board,
        *,
        show_current_probs: bool = True,
        disp_depth: Optional[int] = None,
        disp_topk: Optional[int] = None,
        max_history: int = 6
    ):
        """Print active & recent prophecies. For active, show odds (shared-tree if available, else initial)."""
        now = self.current_ply(board)
        active = self.active
        recent = self.history[-max_history:]

        print("\n== Active prophecies ==" if active else "\n== Active prophecies == (none)")
        for pr in active:
            left = max(0, pr.expires_at_ply - now)
            if show_current_probs and self.reuse.enabled and self.shared_tree is not None and self.shared_tree_root_key == fen_key(board):
                p = self.shared_probs.get(pr.event.key, pr.initial_prob)
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
        Print per-event probabilities from the current position.
        - Reuse shared tree if available; else do a shallow display-only scan.
        """
        use_shared = (
            prefer_shared
            and self.reuse.enabled
            and self.shared_tree is not None
            and self.shared_tree_root_key == fen_key(board)
        )

        probs: Dict[str, float] = {}
        if use_shared:
            for ev in self.events:
                probs[ev.key] = self.shared_probs.get(ev.key, 0.0)
            title = "== Event probabilities (shared tree) =="
        else:
            for ev in self.events:
                p = event_probability(
                    self.engine, board, ev, self.side,
                    plies=min(self.horizon, 6),
                    topk=max(2, disp_topk),
                    depth=max(2, disp_depth),
                    temp=temp or self.temp
                )
                probs[ev.key] = p
            title = "== Event probabilities (display scan) =="

        print("\n" + title)

        def bar(pct: int) -> str:
            filled = min(20, pct // 5)
            return "█" * filled + "·" * (20 - filled)

        for ev in self.events:
            pct = int(round(100 * probs.get(ev.key, 0.0)))
            print(f" • {ev.label:<30} [{bar(pct)}] {pct:3d}%")
