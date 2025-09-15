# Copied from eva_p1_workflow_and_agent.py (depends on many symbols kept intact)
import os, json, time, shutil, pathlib
from typing import Dict, Any, Optional, List
from eva_env_base import log, SYSTEM_BASE_DIR, GPT_AVAILABLE
from eva_p1.comfy_client import ComfyClient
from eva_p1.video_analyzer import VideoAnalyzer
from eva_p1.workflow import validate_workflow_nodes
from eva_p1.multi_bandit import MultiDimensionalBandit
from eva_p1.openrouter_analyzer import OpenRouterAnalyzer
from eva_p1.knowledge_analyzer import KnowledgeAnalyzer
from eva_p1.prompt_generator import MegaEroticJSONPromptGenerator
import cv2

class EnhancedVideoAgentV4:
    """Enhanced Video Agent v4 with Mega Erotic JSON Prompt Generator Integration + GPT"""

    def __init__(self, api: str, base_workflow: str, state_dir: str = None, seconds: float = 5.0, openrouter_key: str = None,
                 reference_only: bool = False, reference_file: Optional[str] = None):
        if state_dir is None:
            state_dir = f"{SYSTEM_BASE_DIR}/auto_state"

        # Initialize paths
        self.state_dir = state_dir
        self.comfyui_output = "/workspace/ComfyUI/output"
        self.review_dir = f"{SYSTEM_BASE_DIR}/video_reviews"
        self.prompts_dir = f"{SYSTEM_BASE_DIR}/generated_prompts"

        # Create directories
        for directory in [self.state_dir, self.review_dir, self.prompts_dir]:
            os.makedirs(directory, exist_ok=True)

        for subdir in ["pending", "rated", "thumbnails"]:
            os.makedirs(os.path.join(self.review_dir, subdir), exist_ok=True)

        # Initialize components
        self.client = ComfyClient(api)

        # Load workflow
        try:
            with open(base_workflow, "r", encoding="utf-8") as f:
                self.base_wf = json.load(f)
        except FileNotFoundError:
            log.error(f"Workflow file not found: {base_workflow}")
            raise
        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in workflow file: {e}")
            raise

        # Validate workflow
        if not validate_workflow_nodes(self.base_wf):
            raise ValueError("Workflow validation failed")

        self.seconds = max(7.0, seconds)
        self.analyzer = VideoAnalyzer()

        # Setup knowledge system
        self.knowledge_path = os.path.join(self.state_dir, "knowledge.json")
        self.ratings_path = os.path.join(self.state_dir, "manual_ratings.json") 
        self.queue_path = os.path.join(self.state_dir, "review_queue.json")

        self.knowledge = self._load_knowledge()
        self.manual_ratings = self._load_manual_ratings()
        self.review_queue = self._load_review_queue()

        # Initialize MEGA EROTIC JSON prompt generator with knowledge analyzer
        self.knowledge_analyzer = KnowledgeAnalyzer(self.knowledge)
        self.mega_erotic_generator = MegaEroticJSONPromptGenerator(self.knowledge_analyzer)

        # Initialize multi-dimensional bandit
        bandit_path = os.path.join(self.state_dir, "bandit_state.json")
        self.bandit = MultiDimensionalBandit(bandit_path)

        # Reference-only mode configuration
        self.reference_only_mode = bool(reference_only)
        self.reference_file = reference_file
        self.reference_params: List[Dict[str, Any]] = []
        if self.reference_only_mode:
            try:
                self.reference_params = self.bandit.load_reference_params(self.reference_file)
                if self.reference_params:
                    log.info(f"🌟 REFERENCE-ONLY MODE: Завантажено {len(self.reference_params)} еталонних комбінацій")
                    for i, p in enumerate(self.reference_params):
                        try:
                            log.info(f"  {i+1}. {self._format_params_info(p)}")
                        except Exception:
                            pass
                else:
                    log.warning("⚠️ reference-only: список порожній — вимикаємо режим")
                    self.reference_only_mode = False
            except Exception as e:
                log.warning(f"Reference-only load failed: {e}")
                self.reference_only_mode = False

        # 🤖 ДОДАЄМО GPT АНАЛІЗАТОР
        if openrouter_key and GPT_AVAILABLE:
            try:
                self.gpt_analyzer = OpenRouterAnalyzer(openrouter_key)
                log.info("🤖 GPT-powered rating analysis enabled")
            except Exception as e:
                log.error(f"Failed to initialize GPT analyzer: {e}")
                self.gpt_analyzer = None
        else:
            self.gpt_analyzer = None
            if openrouter_key and not GPT_AVAILABLE:
                log.warning("⚠️ OpenRouter key provided but OpenAI library not installed")
            else:
                log.info("⚠️ No OpenRouter key provided, using basic analysis")

        log.info(f"✅ Initialized Enhanced Video Agent v4 with MEGA EROTIC JSON Prompt Generator {'+ GPT' if self.gpt_analyzer else ''}")
        log.info(f"📁 ComfyUI output: {self.comfyui_output}")
        log.info(f"📁 System state: {self.state_dir}")
        log.info(f"🎯 Multi-dimensional bandit ready")
        log.info(f"🔥 MEGA EROTIC intelligent JSON prompt generation enabled")
        log.info(f"⏰ Minimum video duration: {self.seconds}s")
        # Duplicate to main logger if available
        try:
            from eva_p3.logger import EnhancedLogger as _EL
            _dummy = _EL
            log.info("🪵 Logging initialized; check auto_state/logs_improved/main.log or logs/main.log")
        except Exception:
            pass

    def get_stats_v4(self) -> Dict[str, Any]:
        """Return aggregated stats for QA CLI and UI.

        Mirrors the structure used by the web server: total_generated, total_rated,
        pending_count, avg_rating, best_score, bandit_iterations, learning_arms.
        """
        try:
            knowledge = self._load_knowledge() if isinstance(self.knowledge, dict) else {}
            manual = self._load_manual_ratings() if isinstance(self.manual_ratings, dict) else {}
            # Bandit state
            bandit_path = os.path.join(self.state_dir, "bandit_state.json")
            bandit_state = {}
            try:
                if os.path.exists(bandit_path):
                    with open(bandit_path, 'r', encoding='utf-8') as f:
                        bandit_state = json.load(f)
            except Exception:
                bandit_state = {}

            total_generated = len(knowledge.get("history", [])) if isinstance(knowledge, dict) else 0
            total_rated = len(manual) if isinstance(manual, dict) else 0

            # Pending = files present but not rated yet
            pending_count = 0
            try:
                import glob
                video_files = glob.glob(f"{self.comfyui_output}/*.mp4")
                rated_names = set(manual.keys()) if isinstance(manual, dict) else set()
                pending_count = len([p for p in video_files if os.path.basename(p) not in rated_names])
            except Exception:
                pending_count = 0

            # Average manual overall
            avg_rating = 0.0
            try:
                vals = []
                for v in (manual.values() if isinstance(manual, dict) else []):
                    r = v.get('rating') if isinstance(v, dict) else None
                    if isinstance(r, dict) and 'overall_quality' in r:
                        vals.append(float(r.get('overall_quality', 0)))
                if vals:
                    avg_rating = sum(vals) / len(vals)
            except Exception:
                avg_rating = 0.0

            return {
                "total_generated": total_generated,
                "total_rated": total_rated,
                "pending_count": pending_count,
                "avg_rating": avg_rating,
                "best_score": (knowledge or {}).get("best_score", 0),
                "bandit_iterations": (bandit_state or {}).get("t", 0),
                "learning_arms": len((bandit_state or {}).get("arms", [])),
            }
        except Exception as e:
            log.warning(f"get_stats_v4 failed: {e}")
            return {
                "total_generated": 0,
                "total_rated": 0,
                "pending_count": 0,
                "avg_rating": 0.0,
                "best_score": 0,
                "bandit_iterations": 0,
                "learning_arms": 0,
            }

    def _extract_manual_overall(self, rating_data: dict) -> float:
        """Універсальний екстрактор ручних оцінок для різних форматів"""
        r = rating_data.get("rating", {})
        val = r.get("overall_quality") or r.get("overall")

        if val is None:
            val = rating_data.get("user_feedback", {}).get("overall_rating")

        if val is None:
            val = r.get("quality") or r.get("score") or rating_data.get("overall")

        try:
            return float(val)
        except (TypeError, ValueError):
            log.warning(f"Cannot parse manual rating value: {val}, using default 5.0")
            return 5.0

    def _load_knowledge(self) -> Dict[str, Any]:
        """Load knowledge database"""
        if os.path.isfile(self.knowledge_path):
            try:
                with open(self.knowledge_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load knowledge: {e}")
        return {"best_score": -1.0, "best_params": {}, "best_combo": None, "history": []}

    def _save_knowledge(self):
        """Save knowledge database with full parameters"""
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, ensure_ascii=False, indent=2)

    def _load_manual_ratings(self) -> Dict[str, Any]:
        """Load manual ratings database"""
        if os.path.isfile(self.ratings_path):
            try:
                with open(self.ratings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load manual ratings: {e}")
        return {}

    def _load_review_queue(self) -> Dict[str, Any]:
        """Load review queue"""
        if os.path.isfile(self.queue_path):
            try:
                with open(self.queue_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load review queue: {e}")
        return {"pending": [], "in_review": [], "completed": []}

    def _save_review_queue(self):
        """Save review queue"""
        with open(self.queue_path, "w", encoding="utf-8") as f:
            json.dump(self.review_queue, f, ensure_ascii=False, indent=2)

    def _check_and_process_new_ratings(self):
        """Перевіряє, чи з'явилися нові manual_ratings і запускає OpenRouter аналіз для них.

        Зберігає результати в state_dir/openrouter_results.json і використовує
        mtime manual_ratings.json, щоб уникати повторної обробки.
        """
        try:
            if not getattr(self, 'gpt_analyzer', None):
                return
            ratings_file = os.path.join(self.state_dir, "manual_ratings.json")
            if not os.path.exists(ratings_file):
                return
            last_flag = os.path.join(self.state_dir, "last_rating_check.txt")
            current_mtime = 0.0
            try:
                current_mtime = os.path.getmtime(ratings_file)
            except Exception:
                return

            last_check_time = 0.0
            if os.path.exists(last_flag):
                try:
                    with open(last_flag, 'r', encoding='utf-8') as f:
                        last_check_time = float((f.read() or '0').strip())
                except Exception:
                    last_check_time = 0.0

            if current_mtime <= last_check_time:
                return

            log.info("🤖 Знайдено нові manual_ratings — запускаємо OpenRouter аналіз")
            try:
                with open(ratings_file, 'r', encoding='utf-8') as f:
                    ratings_data = json.load(f)
            except Exception as e:
                log.warning(f"Не вдалося прочитати manual_ratings.json: {e}")
                return

            analysis_file = os.path.join(self.state_dir, "openrouter_results.json")
            existing = {}
            try:
                if os.path.exists(analysis_file):
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f) or {}
            except Exception:
                existing = {}

            for video_name, rating_data in (ratings_data.items() if isinstance(ratings_data, dict) else []):
                try:
                    result = self.gpt_analyzer.analyze_manual_rating(video_name, rating_data)
                    log.info(f"✅ OpenRouter: {video_name} score={result.get('quality_score', 0):.3f}")
                    existing[video_name] = {
                        "analysis": result,
                        "processed_at": time.time(),
                    }
                except Exception as e:
                    log.warning(f"OpenRouter помилка для {video_name}: {e}")

            try:
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)
            except Exception as e:
                log.warning(f"Не вдалося зберегти openrouter_results.json: {e}")

            try:
                with open(last_flag, 'w', encoding='utf-8') as f:
                    f.write(str(current_mtime))
            except Exception:
                pass
        except Exception as e:
            log.warning(f"_check_and_process_new_ratings failed: {e}")

    def find_generated_video(self, prefix: str) -> Optional[str]:
        """Find generated video in ComfyUI output directory"""
        search_paths = [
            "/workspace/ComfyUI/output",
            "/workspace/ComfyUI/output/video",
            f"{self.comfyui_output}",
            "./workspace/ComfyUI/output",
            "workspace/ComfyUI/output",
            "./output",
            "output",
        ]

        log.info(f"🔍 Searching for video with prefix: {prefix}")

        import time as _t
        newest_any = (None, -1.0)
        for search_dir in search_paths:
            if not os.path.isdir(search_dir):
                continue
            try:
                for entry in os.scandir(search_dir):
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.lower().endswith('.mp4'):
                        continue
                    mtime = entry.stat().st_mtime
                    # Prefer prefix match first
                    if prefix and prefix in name:
                        log.info(f"✅ Found by prefix in {search_dir}: {name}")
                        return os.path.join(search_dir, name)
                    # Track newest as fallback within last 3 minutes
                    if _t.time() - mtime < 180 and mtime > newest_any[1]:
                        newest_any = (os.path.join(search_dir, name), mtime)
            except Exception as e:
                log.warning(f"Error searching in {search_dir}: {e}")

        if newest_any[0]:
            log.info(f"✅ Found recent video: {newest_any[0]}")
            return newest_any[0]
        return None

    def create_thumbnail(self, video_path: str, video_id: str) -> Optional[str]:
        """Create thumbnail from video middle frame"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            mid_frame = frame_count // 2

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            cap.release()

            if ret:
                thumbnail_path = os.path.join(self.review_dir, "thumbnails", f"{video_id}.jpg")
                resized = cv2.resize(frame, (320, 240))
                cv2.imwrite(thumbnail_path, resized)
                return thumbnail_path
        except Exception as e:
            log.error(f"Failed to create thumbnail: {e}")

        return None

    def add_to_review_queue(self, video_path: str, params: Dict[str, Any], auto_metrics: Dict[str, float], combo: List[str]):
        """Add video to manual review queue"""
        import time
        video_id = os.path.basename(video_path).replace('.mp4', '')

        # Create thumbnail
        thumbnail_path = self.create_thumbnail(video_path, video_id)

        # Copy to pending directory
        pending_path = os.path.join(self.review_dir, "pending", f"{video_id}.mp4")
        try:
            shutil.copy2(video_path, pending_path)
        except Exception as e:
            log.warning(f"Failed to copy video to pending: {e}")
            pending_path = video_path

        # Calculate priority based on auto metrics
        score = auto_metrics.get("overall", 0.5)
        if score > 0.8 or score < 0.3:
            priority = 1
        elif score > 0.6:
            priority = 2
        else:
            priority = 3

        # Add to queue
        queue_item = {
            "video_id": video_id,
            "video_path": pending_path,
            "original_path": video_path,
            "thumbnail_path": thumbnail_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "auto_metrics": auto_metrics,
            "prompt": params.get("prompt", ""),
            "params": params,
            "combo": combo,
            "priority": priority
        }

        self.review_queue["pending"].append(queue_item)
        self._save_review_queue()

        log.info(f"✚ Added to review queue: {video_id} (priority: {priority})")

    def _format_params_info(self, params: Dict[str, Any]) -> str:
        """Compact params summary for logs"""
        keys = ['sampler', 'scheduler', 'fps', 'cfg_scale', 'steps', 'width', 'height']
        parts = []
        for k in keys:
            if k in params:
                parts.append(f"{k}={params.get(k)}")
        return " | ".join(parts)

    def generate_next_params(self) -> Dict[str, Any]:
        """Pick next params according to mode (reference-only or bandit)."""
        if self.reference_only_mode and self.reference_params:
            selected = self.bandit.select_reference_only(self.reference_params)
            # Ensure seconds present
            if 'seconds' not in selected:
                selected['seconds'] = self.seconds
            log.info(f"🌟 REFERENCE MODE: {self._format_params_info(selected)}")
            return selected
        # fallback: bandit-driven
        return self.bandit.select_params()

    # ==== High-level search/generation loop expected by QA CLI ====
    def run_iteration_v4(self, params: Dict[str, Any]):
        """Single iteration: apply params to workflow, queue in ComfyUI, wait, analyze, update knowledge/bandit.

        Returns: (score, metrics, video_path, applied_workflow)
        """
        # Перевіряємо наявність нових ручних оцінок і запускаємо OpenRouter для них
        try:
            self._check_and_process_new_ratings()
        except Exception:
            pass
        from eva_p1.workflow import apply_enhanced_params_to_workflow
        # Always autogenerate prompt/negative via generator and IGNORE any prompt from reference params
        # Ensure uniqueness and no carry-over from incoming params
        params.pop('prompt', None)
        params.pop('negative_prompt', None)
        try:
            pj = self.mega_erotic_generator.generate_ultra_detailed_json_prompt()
            gen_text = self.mega_erotic_generator.convert_to_erotic_text_prompt(pj)
            # Append unique token to ensure uniqueness even with similar base prompt
            unique_suffix = f" [id:{int(time.time())}]"
            params['prompt'] = f"{gen_text}{unique_suffix}"

            # Base negative from generator + extended hard blacklist for style/quality/anatomy artifacts
            extended_negative = (
                "anime, manga, cartoon, animated, 2d, illustration, drawing, sketch, painting, artwork, digital art, cgi, "
                "3d render, stylized, cel shading, toon shading, comic style, graphic novel, webtoon, manhwa, chibi, kawaii, "
                "doll, toy, figurine, plastic, artificial, fake, synthetic, rendered, computer generated, video game character, "
                "fantasy character, unreal, surreal, abstract, conceptual, artistic interpretation, low quality, worst quality, "
                "blurry, out of focus, pixelated, compressed, jpeg artifacts, noise, grain, distorted, disfigured, malformed, "
                "mutated, ugly, grotesque, hideous, repulsive, bad anatomy, wrong anatomy, extra limbs, missing limbs, extra arms, "
                "missing arms, extra legs, missing legs, extra fingers, missing fingers, fused fingers, too many fingers, extra hands, "
                "missing hands, malformed hands, bad hands, poorly drawn hands, extra heads, missing head, multiple heads, extra eyes, "
                "missing eyes, cross-eyed, extra mouth, missing mouth, extra nose, bad face, poorly drawn face, asymmetrical face, long neck, "
                "short neck, thick neck, no neck, extra body parts, missing body parts, floating limbs, disconnected limbs, cropped limbs, "
                "cut off, out of frame, text, watermark, signature, logo, username, artist name, copyright, brand, trademark, oversaturated, "
                "undersaturated, overexposed, underexposed, high contrast, low contrast, monochrome when color expected, sepia, black and white "
                "when color expected, static, motionless, frozen, lifeless, robotic, mechanical, stiff, unnatural movement, jerky animation, "
                "low frame rate, stuttering, glitching, artifacting"
            )
            base_neg = self.mega_erotic_generator.get_erotic_negative_prompt(pj)
            params['negative_prompt'] = (base_neg + ", " + extended_negative).strip(', ')
        except Exception as e:
            log.warning(f"Prompt generation failed: {e}")
            # Hard fallback to ensure we NEVER reuse prompt from reference params
            prefix = params.get('prefix') or f"gen_{int(time.time())}"
            params['prompt'] = f"ultra-detailed cinematic realistic scene, high quality, 4k, [id:{prefix}]"
            params['negative_prompt'] = (
                "blurry, low quality, jpeg artifacts, bad anatomy, extra limbs, deformed, watermark, text, logo"
            )
        # Set unique prefix to bind outputs; save prompt artifacts
        prefix = f"gen_{int(time.time())}"
        try:
            params['prefix'] = prefix
            # Save prompt artifacts for traceability
            os.makedirs(self.prompts_dir, exist_ok=True)
            prompt_txt_path = os.path.join(self.prompts_dir, f"{prefix}.prompt.txt")
            with open(prompt_txt_path, 'w', encoding='utf-8') as f:
                f.write(params.get('prompt', ''))
            prompt_json_path = os.path.join(self.prompts_dir, f"{prefix}.prompt.json")
            with open(prompt_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': int(time.time()),
                    'prompt': params.get('prompt'),
                    'negative_prompt': params.get('negative_prompt'),
                    'params': {k: v for k, v in params.items() if k not in ('prompt','negative_prompt')},
                }, f, ensure_ascii=False, indent=2)
            log.info(f"📝 Saved prompt: {prompt_txt_path}")
        except Exception as e:
            log.warning(f"Failed to save prompt artifacts: {e}")

        # Debug-log first 120 chars of the final prompt to show what will be used
        try:
            ptxt = params.get('prompt', '')
            phash = hex(abs(hash(ptxt)) & 0xffffffff)
            log.info(f"🧾 Prompt (hash={phash}): {ptxt[:120]}{'...' if len(ptxt)>120 else ''}")
        except Exception:
            pass

        # Prepare workflow with params
        wf = apply_enhanced_params_to_workflow(self.base_wf, params)
        try:
            # Queue job
            log.info(f"🚀 Генерація: {self._format_params_info(params)} | seconds={params.get('seconds', self.seconds)}")
            prompt_id = self.client.queue(wf)
            hist = self.client.wait(prompt_id, timeout_s=int(max(600, params.get('seconds', self.seconds) * 120)))
        except Exception as e:
            log.warning(f"ComfyUI generation failed: {e}")
            return 0.0, {"error": str(e)}, None, wf

        # Find produced video (by recent timestamp or prefix)
        video_path = self.find_generated_video(prefix) or self.find_generated_video("")
        metrics = {}
        score = 0.0
        if video_path and os.path.exists(video_path):
            # Basic analysis (fast)
            try:
                log.info("🔎 Старт базового аналізу відео")
                metrics = self.analyzer.analyze(video_path)
                score = float(metrics.get('overall', 0.0))
                log.info(f"📈 Результати аналізу: overall={score:.3f}")
            except Exception as e:
                log.warning(f"Basic analysis failed: {e}")
                metrics = {"overall": 0.0}

            # Update knowledge
            try:
                entry = {
                    "video": video_path,
                    "timestamp": int(time.time()),
                    "params": params,
                    "metrics": metrics,
                    "score": score,
                    "prompt": params.get('prompt'),
                    "negative_prompt": params.get('negative_prompt'),
                    "combo": [params.get('sampler'), params.get('scheduler')],
                }
                self.knowledge.setdefault("history", []).append(entry)
                if score > self.knowledge.get("best_score", 0):
                    self.knowledge["best_score"] = score
                    self.knowledge["best_params"] = {"params": params, "metrics": metrics}
                self._save_knowledge()
            except Exception as e:
                log.warning(f"Knowledge update failed: {e}")

            # Add to manual review queue (helps UI)
            try:
                combo = [params.get('sampler'), params.get('scheduler')]
                self.add_to_review_queue(video_path, params, metrics, combo)
            except Exception:
                pass

            # Update bandit
            try:
                self.bandit.update(params, max(0.0, min(1.0, score)))
            except Exception as e:
                log.warning(f"Bandit update failed: {e}")

        return score, metrics, video_path, wf

    def search_v4(self, iterations: int = 10):
        """Main search loop.

        Behavior:
        - If reference-only mode is enabled, iterate only over reference params.
        - Else if whitelist file auto_state/reference_params.json contains combos, iterate over them (cycling).
        - Otherwise, fall back to MultiDimensionalBandit selection.
        """
        iters = int(max(1, iterations))
        # Strict reference-only mode
        if self.reference_only_mode and self.reference_params:
            log.info(f"✅ Reference-only mode активний: {len(self.reference_params)} комбінацій")
            for i in range(iters):
                # В reference‑only режимі інкрементуємо bandit.t у select_reference_only,
                # тому додатково тут t не чіпаємо
                params = self.generate_next_params()
                # При логуванні не показуємо сирі prompt/negative з reference-файлу
                safe = dict(params)
                safe.pop('prompt', None)
                safe.pop('negative_prompt', None)
                log.info(f"▶️ Iteration {i+1}/{iters} (reference-only): params={safe}")
                score, metrics, video_path, _ = self.run_iteration_v4(params)
                log.info(f"✅ Done iter {i+1}: score={score:.3f}, video={video_path}")
            return

        wl = self._load_whitelist_params()
        if wl:
            log.info(f"✅ Whitelist mode: {len(wl)} preset combos found in reference_params.json")
            for i in range(iters):
                raw = wl[i % len(wl)]
                params = raw.get('params') if isinstance(raw, dict) else None
                if not isinstance(params, dict):
                    params = raw if isinstance(raw, dict) else {}
                # Ensure seconds present
                params.setdefault('seconds', self.seconds)
                safe = dict(params)
                safe.pop('prompt', None)
                safe.pop('negative_prompt', None)
                log.info(f"▶️ Iteration {i+1}/{iters} (whitelist): params={safe}")
                score, metrics, video_path, _ = self.run_iteration_v4(params)
                log.info(f"✅ Done iter {i+1}: score={score:.3f}, video={video_path}")
            return

        # Fallback: bandit-driven search
        for i in range(iters):
            try:
                params = self.bandit.select_params()
            except Exception as e:
                log.warning(f"Bandit param select failed: {e}. Using defaults.")
                params = {"fps": 20, "seconds": self.seconds, "sampler": "euler", "scheduler": "normal", "steps": 25, "cfg_scale": 7.0, "width": 768, "height": 432}
            log.info(f"▶️ Iteration {i+1}/{iters}: params={params}")
            score, metrics, video_path, _ = self.run_iteration_v4(params)
            log.info(f"✅ Done iter {i+1}: score={score:.3f}, video={video_path}")

    def _load_whitelist_params(self):
        """Load whitelist parameter combinations from reference_params.json in state_dir.

        Supported formats:
        - Array of {"params": {...}}
        - Array of params dicts
        - {"combos": [...]} or {"params_list": [...]} wrappers
        - {"reference_combinations": [...]} or {"reference_videos": [{"params":{...}}]}
        Returns list (possibly empty) of items (dicts). Caller will extract 'params' if present.
        """
        import json
        ref_path = os.path.join(self.state_dir, 'reference_params.json')
        try:
            if not os.path.exists(ref_path):
                return []
            with open(ref_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                if isinstance(data.get('reference_combinations'), list):
                    return data.get('reference_combinations')
                if isinstance(data.get('reference_videos'), list):
                    # normalize to list of {"params": {...}}
                    out = []
                    for it in data.get('reference_videos'):
                        if isinstance(it, dict):
                            if isinstance(it.get('params'), dict):
                                out.append({'params': it.get('params')})
                            else:
                                out.append(it)
                    return out
                for key in ('combos', 'params_list', 'list'):
                    if isinstance(data.get(key), list):
                        return data.get(key)
            return []
        except Exception as e:
            log.warning(f"Failed to load whitelist params: {e}")
            return []

