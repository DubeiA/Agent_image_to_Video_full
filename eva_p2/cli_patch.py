# 1:1 copy from eva_p2_cli_patch.py (top section)
def _patch_cli_for_merged_agent(original_code_marker: str = "Enhanced Video Agent v4 with MEGA EROTIC JSON Prompt Generator + GPT"):
    # This method is a no-op at runtime; we keep it for source clarity.
    return True
# =============================================================================

def main():
    ensure_logs_directory()
    parser = argparse.ArgumentParser(description="Enhanced Video Agent v4 with MEGA EROTIC JSON Prompt Generator + GPT")
    parser.add_argument("--api", default="http://127.0.0.1:8188", help="ComfyUI API URL")
    parser.add_argument("--workflow", default="/workspace/wan22_system/video_wan2_2_14B_t2v.json", help="Workflow JSON file")
    parser.add_argument("--state-dir", default="/workspace/wan22_system/auto_state", help="State directory")
    parser.add_argument("--seconds", type=float, default=5.0, help="Video duration (minimum 5s)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--openrouter-key", help="OpenRouter API key for GPT analysis (optional)")  # 🤖 НОВИЙ АРГУМЕНТ

    
    parser.add_argument("--use-enhanced-analysis", action="store_true", help="Use improved analyzer in addition to base metrics")
    parser.add_argument("--train-improved", action="store_true", help="Run improved training system first")
    args = parser.parse_args()

    if not os.path.exists(args.workflow):
        log.error(f"❌ Workflow file not found: {args.workflow}")
        sys.exit(1)

    try:
        response = requests.get(f"{args.api}/system_stats", timeout=5)
        if response.status_code != 200:
            log.warning("⚠️ ComfyUI may not be running or accessible")
    except Exception:
        log.warning("⚠️ Cannot connect to ComfyUI API")

    log.info(f"🚀 Starting Enhanced Video Agent v4 with MEGA EROTIC JSON Prompt Generator + GPT")
    log.info(f"📍 Workflow: {args.workflow}")
    log.info(f"📍 State dir: {args.state_dir}")
    log.info(f"📍 ComfyUI API: {args.api}")
    log.info(f"🎬 Video duration: {max(5.0, args.seconds)}s (enforced minimum 5s)")
    log.info(f"🔄 Iterations: {args.iterations}")
    log.info(f"🔥 MEGA EROTIC content generation enabled")
    log.info(f"🤖 GPT analysis: {'enabled' if args.openrouter_key else 'disabled'}")

    try:
        agent = EnhancedVideoAgentV4Merged(
            api=args.api,
            base_workflow=args.workflow,
            state_dir=args.state_dir,
            seconds=max(5.0, args.seconds),
            openrouter_key=args.openrouter_key  # 🤖 ДОДАЄМО GPT
        )

        stats = agent.get_stats_v4()
        log.info(f"📊 Initial stats: {stats}")

        agent.search_v4(iterations=args.iterations)

        final_stats = agent.get_stats_v4()
        log.info(f"🏁 Final MEGA EROTIC stats: {final_stats}")

    except KeyboardInterrupt:
        log.info("⏹️ MEGA EROTIC generation stopped by user")
    except Exception as e:
        log.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
