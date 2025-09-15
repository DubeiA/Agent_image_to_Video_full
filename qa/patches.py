import os
from qa.agent_namespace import agent_mod

def patch_logging_to_auto_state():
    base_dir = os.environ.get("WORKSPACE_DIR", "/workspace/wan22_system/")
    logs_dir = os.path.join(base_dir, "auto_state", "logs_improved")
    os.makedirs(logs_dir, exist_ok=True)

    if hasattr(agent_mod, "EnhancedLogger"):
        OriginalLogger = agent_mod.EnhancedLogger

        import logging as _logging
        try:
            root = _logging.getLogger()
            root.handlers = []
            root.propagate = False
        except Exception:
            pass
        try:
            module_log = getattr(agent_mod, 'log', None)
            if module_log is not None:
                module_log.handlers = []
                module_log.propagate = False
        except Exception:
            pass

        class PatchedLogger(OriginalLogger):
            def __init__(self, config, log_dir=None):
                self._qa_log_dir = logs_dir
                self.log_dir = logs_dir
                super().__init__(config)

            def setup_loggers(self):
                import logging, sys, datetime, os as _os
                _os.makedirs(os.path.join(base_dir, "auto_state", "logs_improved"), exist_ok=True)

                self.main_logger = logging.getLogger("enhanced_video_agent")
                self.main_logger.setLevel(getattr(logging, self.config.log_level))

                main_fp = os.path.join(base_dir, "auto_state", "logs_improved", "main.log")
                training_fp = os.path.join(base_dir, "auto_state", "logs_improved", "training.log")
                analysis_fp = os.path.join(base_dir, "auto_state", "logs_improved", "analysis.log")

                main_handler = logging.FileHandler(main_fp)
                training_handler = logging.FileHandler(training_fp)
                analysis_handler = logging.FileHandler(analysis_fp)
                console_handler = logging.StreamHandler(sys.stdout)

                detailed_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
                simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

                main_handler.setFormatter(detailed_formatter)
                training_handler.setFormatter(detailed_formatter)
                analysis_handler.setFormatter(detailed_formatter)
                console_handler.setFormatter(simple_formatter)

                self.main_logger.handlers = []
                self.main_logger.addHandler(main_handler)
                self.main_logger.addHandler(console_handler)
                self.main_logger.propagate = False

                # Also route module-level logger (agent_mod.log from eva_env_base)
                try:
                    module_log = getattr(agent_mod, 'log', None)
                    if module_log is not None:
                        module_log.setLevel(getattr(logging, self.config.log_level))
                        module_log.handlers = []
                        module_log.addHandler(main_handler)
                        module_log.addHandler(console_handler)
                        module_log.propagate = False
                except Exception:
                    pass

                self.training_logger = logging.getLogger("training")
                self.training_logger.setLevel(logging.DEBUG)
                self.training_logger.handlers = []
                self.training_logger.addHandler(training_handler)
                self.training_logger.propagate = False

                self.analysis_logger = logging.getLogger("analysis")
                self.analysis_logger.setLevel(logging.DEBUG)
                self.analysis_logger.handlers = []
                self.analysis_logger.addHandler(analysis_handler)
                self.analysis_logger.propagate = False

                class _DedupFilter(logging.Filter):
                    def __init__(self):
                        super().__init__(name="dedup")
                        self._last = None
                    def filter(self, record):
                        msg = record.getMessage()
                        if msg == self._last:
                            return False
                        self._last = msg
                        return True

                class _OptFlowFilter(logging.Filter):
                    def __init__(self):
                        super().__init__(name="optflow")
                        self._count = 0
                    def filter(self, record):
                        msg = record.getMessage()
                        if "Error calculating optical flow" in msg:
                            self._count += 1
                            return (self._count % 10) == 1
                        return True

                console_handler.addFilter(_DedupFilter())
                console_handler.addFilter(_OptFlowFilter())

            def _write_jsonl(self, path, obj):
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                except Exception:
                    pass
                try:
                    import json
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        agent_mod.EnhancedLogger = PatchedLogger


def patch_bandit_ban_rule():
    Bandit = agent_mod.MultiDimensionalBandit
    original = Bandit._check_and_ban_poor_combos

    def patched(self):
        newly_banned = 0
        for combo_key, stats_dict in self.combo_stats.items():
            if combo_key in self.banned_combos:
                continue
            scores = list(stats_dict.get("scores", []))
            if len(scores) >= 3:
                last3 = scores[-3:]
                if all(s < 0.55 for s in last3):
                    self.banned_combos.add(combo_key)
                    newly_banned += 1
                    agent_mod.log.info(
                        f"🚫 QA ban rule: last3<{0.55} -> banned {combo_key} | last3={','.join(f'{s:.3f}' for s in last3)}"
                    )
                    continue
        res = original(self)
        if res is None and newly_banned > 0:
            agent_mod.log.info(f"🧹 QA ban rule banned {newly_banned} combos (in addition to base checks)")
        return res

    Bandit._check_and_ban_poor_combos = patched


def patch_video_processor_init():
    try:
        OriginalVP = agent_mod.VideoProcessor
        original_init = OriginalVP.__init__
        def wrapped_init(self, config, logger=None, *args, **kwargs):
            return original_init(self, config)
        OriginalVP.__init__ = wrapped_init
    except Exception:
        pass


def patch_openrouter_quality_guard():
    try:
        ORA = agent_mod.OpenRouterAnalyzer
        original = ORA.analyze_manual_rating
        def wrapped(self, video_name: str, rating_data: dict):
            result = original(self, video_name, rating_data)
            try:
                qs = result.get('quality_score', None)
                if qs is None or (isinstance(qs, float) and (qs != qs)):
                    rating = (rating_data or {}).get('rating', {})
                    overall = rating.get('overall_quality')
                    if isinstance(overall, (int, float)):
                        result['quality_score'] = max(0.0, min(1.0, (float(overall) - 1.0) / 9.0))
                    else:
                        result['quality_score'] = 0.5
            except Exception:
                result['quality_score'] = 0.5
            return result
        ORA.analyze_manual_rating = wrapped
    except Exception:
        pass


def patch_enrich_run_iteration_metrics():
    try:
        EV = agent_mod.EnhancedVideoAgentV4Merged
        original = EV.run_iteration_v4
        def wrapped(self, params):
            result = original(self, params)
            try:
                score, metrics, video_path, wf = result
            except Exception:
                return result
            try:
                if getattr(self, 'video_processor', None) and video_path and os.path.exists(video_path):
                    det = self.video_processor.analyze_video(video_path)
                    det_dict = getattr(det, '__dict__', None)
                    if det_dict is None:
                        try:
                            from dataclasses import asdict as _asdict
                            det_dict = _asdict(det)
                        except Exception:
                            det_dict = {}
                    summary = {}
                    for k in [
                        'confidence','anatomical_score','face_quality_score','artifact_score',
                        'temporal_consistency','face_landmarks_consistency','eye_blink_naturalness',
                        'micro_expression_analysis'
                    ]:
                        v = det_dict.get(k)
                        if isinstance(v, (int,float)):
                            summary[k] = float(v)
                    if isinstance(metrics, dict):
                        metrics.setdefault('enhanced_summary', {}).update(summary)
                        metrics.setdefault('enhanced_details', {}).update({
                            'frames_analyzed': det_dict.get('frames_analyzed', 0),
                            'faces_detected': det_dict.get('faces_detected', 0),
                            'analysis_methods': det_dict.get('analysis_methods', []),
                            'detected_errors': det_dict.get('detected_errors', []),
                        })
            except Exception:
                pass
            return score, metrics, video_path, wf
        EV.run_iteration_v4 = wrapped
    except Exception:
        pass


