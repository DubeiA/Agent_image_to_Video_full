import os
import argparse
from qa.agent_namespace import agent_mod
from qa.patches import (
    patch_logging_to_auto_state,
    patch_bandit_ban_rule,
    patch_video_processor_init,
    patch_openrouter_quality_guard,
    patch_enrich_run_iteration_metrics,
)
import eva_p3.logger as _p3a
import eva_p1.analysis_config as _p1a
try:
    from qa.t2i2v_runner import run_t2i2v
except Exception:
    run_t2i2v = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", required=False)
    parser.add_argument("--state-dir", default="/workspace/wan22_system/auto_state")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--openrouter-key")
    parser.add_argument("--use-enhanced-analysis", action="store_true")
    parser.add_argument("--train-improved", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Використовувати тільки еталонні параметри з reference_params.json")
    parser.add_argument("--reference-file", type=str, help="Шлях до reference_params.json (опційно)")

    # Two-stage (T2I -> I2V) pipeline options
    parser.add_argument("--two-stage", action="store_true", help="Запуск двоетапного T2I→I2V пайплайна")
    parser.add_argument("--t2i2v-root", default="/workspace/Agent_T2I2V", help="Коренева папка ізоляції артефактів/стейту")
    parser.add_argument("--image-workflow", type=str, help="Шлях до T2I workflow (flux_dev_full_text_to_image.json)")
    parser.add_argument("--i2v-workflow", type=str, help="Шлях до I2V workflow (video_wan2_2_14B_i2v.json)")
    parser.add_argument("--image-width", type=int, default=960)
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--i2v-widths", type=str, default="960x540,1280x720,768x432", help="Список роздільностей 16:9 для I2V через кому WxH")
    parser.add_argument("--i2v-fps", type=int, default=20)
    parser.add_argument("--i2v-seconds", type=float, default=6.0)
    parser.add_argument("--randomize-sizes", action="store_true", help="Вибирати роздільність випадково на кожній ітерації")
    parser.add_argument("--randomize-fps", action="store_true", help="Вибирати fps випадково на кожній ітерації")
    parser.add_argument("--fps-min", type=int, default=20)
    parser.add_argument("--fps-max", type=int, default=35)
    parser.add_argument("--simple-prompt-test", action="store_true", help="Використати дуже простий T2I промпт для діагностики")
    args = parser.parse_args()

    if args.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_key

    # Route logs to isolated auto_state if two-stage root provided
    if args.two_stage and args.t2i2v_root:
        os.environ["WORKSPACE_DIR"] = args.t2i2v_root
        os.environ["WAN22_SYSTEM_DIR"] = args.t2i2v_root
        os.makedirs(os.path.join(args.t2i2v_root, "auto_state", "logs_improved"), exist_ok=True)
        # also redirect state-dir by default if user didn't override
        if args.state_dir == "/workspace/wan22_system/auto_state":
            args.state_dir = os.path.join(args.t2i2v_root, "auto_state")

    patch_logging_to_auto_state()
    patch_bandit_ban_rule()
    patch_video_processor_init()
    patch_openrouter_quality_guard()
    patch_enrich_run_iteration_metrics()

    # Two-stage runner branch
    if args.two_stage:
        if run_t2i2v is None:
            raise RuntimeError("Two-stage runner is unavailable (import failed)")
        run_t2i2v(args)
        return

    # Single-stage (legacy) merged agent branch
    if not args.workflow:
        raise SystemExit("--workflow обов'язковий у звичайному режимі (без --two-stage)")

    # ВАЖЛИВО: використовуємо пропатчений логер з agent_mod, щоб мати JSONL та лог-директорію
    logger = agent_mod.EnhancedLogger(_p1a.AnalysisConfig())

    agent = agent_mod.EnhancedVideoAgentV4Merged(
        api=args.api,
        base_workflow=args.workflow,
        state_dir=args.state_dir,
        seconds=max(5.0, args.seconds),
        logger=logger,
        openrouter_key=args.openrouter_key,
        reference_only=bool(args.reference_only),
        reference_file=args.reference_file,
    )

    stats = agent.get_stats_v4()
    agent_mod.log.info(f"📊 QA initial stats: {stats}")
    agent.search_v4(iterations=args.iterations)


