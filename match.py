# match.py
import argparse
import time
import chess
from engine_utils import make_engine, play_best
from events import EASY_EVENTS, EVENTS
from taunter import Taunter, SteeringCfg, ReuseCfg
from policy_tree import TreeParams

# python match.py --reuse --easy-events --easy-events --taunt-horizon 6 --avoid-retarget-plies 6 --scan-every-fullmoves 1 --show-event-probs --plies 6 --root-depth 12 --inner-depth 10 --root-topk 4 --inner-topk 3 --strong-as white --strong-depth 16 --weak-depth 8 --weak-skill 2 --unicode-board --debug

def main():
    ap = argparse.ArgumentParser(description="Strong Stockfish vs Weak Stockfish with taunts + steering + reuse/hybrid.")
    # ap.add_argument("--strong-as", choices=["white", "black"], default="white")
    # ap.add_argument("--strong-depth", type=int, default=16)
    # ap.add_argument("--weak-depth", type=int, default=8)
    # ap.add_argument("--weak-skill", type=int, default=2)
    # # Prophecy analysis params (legacy)
    # ap.add_argument("--taunt-horizon", type=int, default=6)
    # ap.add_argument("--taunt-thresh", type=float, default=0.35)
    # ap.add_argument("--pol-depth", type=int, default=12)
    # ap.add_argument("--topk", type=int, default=4)
    # ap.add_argument("--temp", type=float, default=1.0)
    # # Steering params
    # ap.add_argument("--lam", type=float, default=25.0)
    # ap.add_argument("--trust-drop", type=int, default=30)
    # ap.add_argument("--cand-topk", type=int, default=4)
    # ap.add_argument("--tree-topk", type=int, default=3)
    # ap.add_argument("--tree-depth", type=int, default=12)
    # ap.add_argument("--micro-deepen", action="store_true")
    # # Reuse / shared-tree params
    # ap.add_argument("--reuse", action="store_true", help="Enable shared-tree reuse + motif-aware widening.")
    # ap.add_argument("--plies", type=int, default=6)
    # ap.add_argument("--root-topk", type=int, default=4)
    # ap.add_argument("--inner-topk", type=int, default=3)
    # ap.add_argument("--root-depth", type=int, default=12)
    # ap.add_argument("--inner-depth", type=int, default=10)
    # ap.add_argument("--widen-checks", type=int, default=4)
    # ap.add_argument("--widen-captures", type=int, default=4)
    # ap.add_argument("--widen-hooks", type=int, default=2)
    # ap.add_argument("--widen-lifts", type=int, default=2)
    # ap.add_argument("--widen-cap", type=int, default=6)
    # ap.add_argument("--quick-eval-depth", type=int, default=6)
    # # Display & scheduling
    # ap.add_argument("--unicode-board", action="store_true")
    # ap.add_argument("--show-probs", action="store_true")
    # ap.add_argument("--disp-depth", type=int, default=10)
    # ap.add_argument("--disp-topk", type=int, default=3)
    # ap.add_argument("--max-moves", type=int, default=200)
    # ap.add_argument("--scan-every-fullmoves", type=int, default=1)
    # ap.add_argument("--scan-every-plies", type=int, default=None)
    # ap.add_argument("--respect-scan-schedule-for-steering", action="store_true")
    # ap.add_argument("--debug", action="store_true")

    # ap.add_argument("--show-event-probs", action="store_true", help="Show a per-event probability panel each ply.")
    # ap.add_argument("--event-disp-depth", type=int, default=8, help="Depth for display-only event probabilities.")
    # ap.add_argument("--event-disp-topk", type=int, default=3, help="Top-K branching for display-only event probabilities.")

    ap.add_argument(
        "--strong-as", choices=["white", "black"], default="white",
        help="Which color the strong (taunting) engine plays as."
    )
    ap.add_argument(
        "--strong-depth", type=int, default=16,
        help="Search depth (in plies / half-moves) for the strong engine when choosing its move."
    )
    ap.add_argument(
        "--weak-depth", type=int, default=8,
        help="Search depth (in plies) for the weak engine when choosing its move."
    )
    ap.add_argument(
        "--weak-skill", type=int, default=2,
        help="Stockfish 'Skill Level' for the weak engine (0–20). Lower = weaker, more blunders."
    )

    # Prophecy analysis params (legacy per-event scanning)
    ap.add_argument(
        "--taunt-horizon", type=int, default=6,
        help="Prophecy window length in plies. If the event doesn't occur by this horizon, it’s marked averted."
    )
    ap.add_argument(
        "--taunt-thresh", type=float, default=0.35,
        help="Minimum probability [0–1] required to announce a prophecy."
    )
    ap.add_argument(
        "--pol-depth", type=int, default=12,
        help="Analysis depth (plies) per node for the legacy probability tree (when not using the shared tree)."
    )
    ap.add_argument(
        "--topk", type=int, default=4,
        help="Branching factor per node for the legacy probability tree (top-K moves kept)."
    )
    ap.add_argument(
        "--temp", type=float, default=1.0,
        help="Softmax temperature (in pawns) for converting evals to move weights. Lower = sharper policy."
    )
    ap.add_argument(
        "--easy-events", action="store_true",
        help="Use a higher-frequency test set of events (checks, pawn captures, center entry, etc.)."
    )
    ap.add_argument(
    "--avoid-retarget-plies", type=int, default=None,
    help="If set, once an event is fulfilled/averted, do not target the same event again for this many plies. "
         "Defaults to the prophecy horizon if omitted."
)



    # Steering params
    ap.add_argument(
        "--lam", type=float, default=25.0,
        help="Steering strength: centipawn bonus added to a move's score per +1.0 event probability."
    )
    ap.add_argument(
        "--trust-drop", type=int, default=30,
        help="Trust region (cp). Do not take a steered move if it's worse than the best move by more than this."
    )
    ap.add_argument(
        "--cand-topk", type=int, default=4,
        help="How many candidate root moves the strong engine considers for steering."
    )
    ap.add_argument(
        "--tree-topk", type=int, default=3,
        help="Branching factor for steering-time probability evals (when not using the shared tree)."
    )
    ap.add_argument(
        "--tree-depth", type=int, default=12,
        help="Depth (plies) for steering-time probability evals (when not using the shared tree)."
    )
    ap.add_argument(
        "--micro-deepen", action="store_true",
        help="Enable tiny, targeted re-search on the top steered candidates to refine probabilities."
    )

    # Reuse / shared-tree params
    ap.add_argument(
        "--reuse", action="store_true",
        help="Enable the shared look-ahead tree with motif-aware widening (fast multi-event odds + reuse for steering)."
    )
    ap.add_argument(
        "--plies", type=int, default=6,
        help="Tree horizon in plies for the shared look-ahead tree."
    )
    ap.add_argument(
        "--root-topk", type=int, default=4,
        help="Root-node branching factor (top-K) for the shared tree before widening."
    )
    ap.add_argument(
        "--inner-topk", type=int, default=3,
        help="Inner-node branching factor (top-K) for the shared tree before widening."
    )
    ap.add_argument(
        "--root-depth", type=int, default=12,
        help="Engine depth (plies) at the root when building the shared tree."
    )
    ap.add_argument(
        "--inner-depth", type=int, default=10,
        help="Engine depth (plies) at inner nodes when building the shared tree."
    )
    ap.add_argument(
        "--widen-checks", type=int, default=4,
        help="Motif widening: add up to this many extra checking moves per node."
    )
    ap.add_argument(
        "--widen-captures", type=int, default=4,
        help="Motif widening: add up to this many extra capture moves (esp. majors/near-king) per node."
    )
    ap.add_argument(
        "--widen-hooks", type=int, default=2,
        help="Motif widening: add up to this many extra 'hook' pawn pushes (g/h-file storms) per node."
    )
    ap.add_argument(
        "--widen-lifts", type=int, default=2,
        help="Motif widening: add up to this many extra rook-lift moves (e.g., Rh3/Ra3 paths) per node."
    )
    ap.add_argument(
        "--widen-cap", type=int, default=6,
        help="Cap on the number of widened (extra) moves kept per node in addition to the base top-K."
    )
    ap.add_argument(
        "--quick-eval-depth", type=int, default=6,
        help="Depth (plies) used to quickly score widened moves that weren't in the engine's MultiPV."
    )

    # Display & scheduling
    ap.add_argument(
        "--unicode-board", action="store_true",
        help="Print the board with Unicode pieces and borders instead of ASCII."
    )
    ap.add_argument(
        "--show-probs", action="store_true",
        help="Recompute and show current odds for ACTIVE prophecies each ply (slower)."
    )
    ap.add_argument(
        "--disp-depth", type=int, default=10,
        help="Depth (plies) for display-only prophecy odds when --show-probs is set."
    )
    ap.add_argument(
        "--disp-topk", type=int, default=3,
        help="Top-K branching for display-only prophecy odds when --show-probs is set."
    )
    ap.add_argument(
        "--max-moves", type=int, default=200,
        help="Safety cap on half-moves (plies) before the game auto-stops."
    )
    ap.add_argument(
        "--scan-every-fullmoves", type=int, default=1,
        help="Run the prophecy scan/taunt every K FULL moves (1 = every move pair)."
    )
    ap.add_argument(
        "--scan-every-plies", type=int, default=None,
        help="Run the prophecy scan/taunt every K plies (half-moves). If set, overrides --scan-every-fullmoves."
    )
    ap.add_argument(
        "--respect-scan-schedule-for-steering", action="store_true",
        help="If set, steering will NOT perform extra scans; it uses only cached targets on non-scan turns."
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Show progress bars and debug markers during scans/steering."
    )

    # Event probability panel
    ap.add_argument(
        "--show-event-probs", action="store_true",
        help="Show a per-event probability panel each ply."
    )
    ap.add_argument(
        "--event-disp-depth", type=int, default=8,
        help="Depth (plies) for display-only per-event probabilities (used if no shared tree to reuse)."
    )
    ap.add_argument(
        "--event-disp-topk", type=int, default=3,
        help="Top-K branching for display-only per-event probabilities (used if no shared tree to reuse)."
    )


    args = ap.parse_args()

    strong_is_white = args.strong_as == "white"

    print("🔧 Launching engines...", flush=True)
    strong_engine = make_engine(skill_level=20, hash_mb=256)
    weak_engine   = make_engine(skill_level=args.weak_skill, hash_mb=64)
    print("✅ Engines ready.", flush=True)

    board = chess.Board()

    reuse_cfg = ReuseCfg(
        enabled=args.reuse,
        params=TreeParams(
            plies=args.plies,
            depth_root=args.root_depth,
            depth_inner=args.inner_depth,
            topk_root=args.root_topk,
            topk_inner=args.inner_topk,
            temp_pawns=args.temp,
            widen_checks=args.widen_checks,
            widen_captures=args.widen_captures,
            widen_hooks=args.widen_hooks,
            widen_lifts=args.widen_lifts,
            widen_total_cap=args.widen_cap,
            quick_eval_depth=args.quick_eval_depth,
        ),
    )

    # taunter = Taunter(
    #     engine=strong_engine,
    #     side=chess.WHITE if strong_is_white else chess.BLACK,
    #     prob_threshold=args.taunt_thresh,
    #     horizon_plies=args.taunt_horizon,
    #     topk=args.topk,
    #     depth=args.pol_depth,
    #     temp=args.temp,
    #     cooldown_plies=6,
    #     debug=args.debug,
    #     reuse=reuse_cfg,
    # )

    events_set = EASY_EVENTS if args.easy_events else EVENTS  # (use the actual Python name: args.easy_events)

    taunter = Taunter(
        engine=strong_engine,
        side=chess.WHITE if strong_is_white else chess.BLACK,
        prob_threshold=args.taunt_thresh,
        horizon_plies=args.taunt_horizon,
        topk=args.topk,
        depth=args.pol_depth,
        temp=args.temp,
        cooldown_plies=6,
        debug=args.debug,
        reuse=reuse_cfg,
        events=events_set,
        avoid_retarget_plies=args.avoid_retarget_plies,  # ← NEW
    )

    steer_cfg = SteeringCfg(
        lam_cp=args.lam,
        trust_region_drop_cp=args.trust_drop,
        candidate_topk=args.cand_topk,
        tree_topk=args.tree_topk,
        tree_depth=args.tree_depth,
        horizon_plies=args.taunt_horizon,
        temp_pawns=args.temp,
        micro_deepen=args.micro_deepen,
    )

    print("=== Taunting Stockfish Match (Reuse/Hybrid) ===")
    print(f"Strong: {args.strong_as.upper()}  (root depth {args.strong_depth})")
    print(f"Weak:   {('black' if strong_is_white else 'white').upper()}  (depth {args.weak_depth}, skill {args.weak_skill})")
    print()

    ply_count = 0
    try:
        while not board.is_game_over() and ply_count < args.max_moves:
            is_strong_turn = board.turn == (chess.WHITE if strong_is_white else chess.BLACK)

            if is_strong_turn:
                taunter.scheduled_scan_and_maybe_taunt(
                    board,
                    every_plies=args.scan_every_plies,
                    every_fullmoves=(args.scan_every_fullmoves if args.scan_every_plies is None else None),
                )

            prev = board.copy()

            if is_strong_turn:
                scan_due_now = taunter.is_scan_due(
                    board,
                    every_plies=args.scan_every_plies,
                    every_fullmoves=(args.scan_every_fullmoves if args.scan_every_plies is None else None),
                )
                mv = taunter.choose_move_with_steering(
                    board,
                    base_depth=args.strong_depth,
                    cfg=steer_cfg,
                    allow_scan=(not args.respect_scan_schedule_for_steering) or scan_due_now
                )
            else:
                mv = play_best(weak_engine, board, depth=args.weak_depth)

            san = board.san(mv)
            board.push(mv)
            ply_count += 1

            # Display
            move_label = f"{board.fullmove_number:>3}{'.' if board.turn == chess.BLACK else '...'} {san}"
            print("\n" + "=" * len(move_label))
            print(move_label)
            print("=" * len(move_label))
            if args.unicode_board:
                try: print(board.unicode(borders=True))
                except Exception: print(board)
            else:
                print(board)

            taunter.update_after_move(prev, mv, board)
            taunter.render_status(
                board,
                show_current_probs=args.show_probs,
                disp_depth=args.disp_depth,
                disp_topk=args.disp_topk,
                max_history=6
            )

            if args.show_event_probs:
                # Prefer the shared tree when reuse is enabled; otherwise do a shallow display scan
                taunter.render_event_probabilities(
                    board,
                    prefer_shared=args.reuse,
                    disp_depth=args.event_disp_depth,
                    disp_topk=args.event_disp_topk,
                    temp=args.temp,
                )

            time.sleep(0.10)

        print("\n=== Game Over ===")
        print("Result:", board.result(), "-", board.outcome())
        print(board)
    finally:
        strong_engine.quit()
        weak_engine.quit()

if __name__ == "__main__":
    main()
