# Copied from eva_p1_comfy_video_bandit.py
import os, json, math, random
from typing import Dict, Any, List
from eva_env_base import log
from eva_p1.analysis_config import FPS_OPTIONS, SECONDS_OPTIONS, CFG_SCALES, STEPS_OPTIONS, RESOLUTION_OPTIONS

class MultiDimensionalBandit:
    """Multi-dimensional UCB bandit with intelligent combo filtering and data migration"""

    def __init__(self, state_path: str):
        self.state_path = state_path
        self.combo_stats = {}  # {combo_key: {"N": int, "S": float, "scores": []}}
        self.t = 0
        self.min_attempts = 3
        self.poor_threshold = 0.45
        self.banned_combos = set()
        self.load()

    def _migrate_old_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old bandit format to new format"""
        migrated = False
        combo_stats = data.get("combo_stats", {})
        
        for combo_key, stats_dict in combo_stats.items():
            # Check if this is old format
            if "total_reward" in stats_dict and "count" in stats_dict:
                # Convert old format to new
                total_reward = stats_dict["total_reward"]
                count = stats_dict["count"]
                
                # Create new format
                new_stats = {
                    "N": count,
                    "S": total_reward,
                    "scores": [total_reward / count] * count  # Approximate scores list
                }
                
                combo_stats[combo_key] = new_stats
                migrated = True
                log.info(f"🔄 Migrated old bandit entry: {combo_key}")
            
            # Ensure all required fields exist
            elif "N" not in stats_dict or "S" not in stats_dict:
                # Fix incomplete entries
                combo_stats[combo_key] = {
                    "N": stats_dict.get("N", stats_dict.get("count", 0)),
                    "S": stats_dict.get("S", stats_dict.get("total_reward", 0.0)),
                    "scores": stats_dict.get("scores", [])
                }
                migrated = True
                log.info(f"🔧 Fixed incomplete bandit entry: {combo_key}")
        
        if migrated:
            data["combo_stats"] = combo_stats
            log.info("💾 Saving migrated bandit state...")
            self._save_data(data)
        
        return data

    def _save_data(self, data: Dict[str, Any]):
        """Save data to file atomically to avoid corruption on crash."""
        try:
            tmp_path = f"{self.state_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.state_path)
        except Exception as e:
            log.error(f"Failed to save bandit state: {e}")

    def _combo_key(self, params: Dict[str, Any]) -> str:
        """Generate unique key for parameter combination"""
        key_parts = [
            params.get("sampler", "unknown"),
            params.get("scheduler", "unknown"),
            str(params.get("fps", 20)),
            str(params.get("cfg_scale", 7.0)),
            str(params.get("steps", 25)),
            f"{params.get('width', 768)}x{params.get('height', 432)}"
        ]
        return "|".join(key_parts)

    def load(self):
        """Load bandit state from file with migration support"""
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Migrate old format if needed
                data = self._migrate_old_format(data)
                
                self.combo_stats = data.get("combo_stats", {})
                self.t = data.get("t", 0)
                self.banned_combos = set(data.get("banned_combos", []))
                
                log.info(f"✅ Loaded multi-dim bandit: t={self.t}, combos={len(self.combo_stats)}, banned={len(self.banned_combos)}")
            except Exception as e:
                log.warning(f"Failed to load bandit state: {e}")
                # Initialize with empty state
                self.combo_stats = {}
                self.t = 0
                self.banned_combos = set()

    def save(self):
        """Save bandit state to file"""
        data = {
            "combo_stats": self.combo_stats,
            "t": self.t,
            "banned_combos": list(self.banned_combos)
        }
        self._save_data(data)

    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameter combination with FIXED PAIRS"""

        # ФІКСОВАНІ ПАРИ SAMPLER + SCHEDULER (ОПТИМІЗОВАНО)
        sampler_scheduler_pairs = [
            ("euler", "simple"), 
            ("euler", "normal"), 
            ("dpmpp_2m", "normal"),
            ("dpmpp_2m_sde", "normal"), 
            ("dpmpp_2m_sde", "simple"),
            ("dpm_2", "normal"),
            ("dpm_2_ancestral", "normal"),
            ("dpmpp_2s_ancestral", "karras"),
            ("dpmpp_sde", "karras")
        ]
        
        sampler, scheduler = random.choice(sampler_scheduler_pairs)
        resolution = random.choice(RESOLUTION_OPTIONS)
        
        return {
            "sampler": sampler,
            "scheduler": scheduler,
            "fps": random.choice(FPS_OPTIONS),
            "seconds": random.choice(SECONDS_OPTIONS),
            "width": resolution[0],
            "height": resolution[1],
            "cfg_scale": random.choice(CFG_SCALES),
            "steps": random.choice(STEPS_OPTIONS)
        }

    def _check_and_ban_poor_combos(self):
        """Check and ban consistently poor performing combinations"""
        newly_banned = 0
        for combo_key, stats_dict in self.combo_stats.items():
            if combo_key in self.banned_combos:
                continue

            N = stats_dict.get("N", 0)
            scores = stats_dict.get("scores", [])

            if N >= self.min_attempts and len(scores) >= self.min_attempts:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)

                if avg_score < self.poor_threshold and max_score < self.poor_threshold + 0.1:
                    self.banned_combos.add(combo_key)
                    newly_banned += 1
                    log.info(f"🚫 Banned poor combo: {combo_key} (avg={avg_score:.3f}, max={max_score:.3f})")

        if newly_banned > 0:
            log.info(f"🧹 Cleaned {newly_banned} poor combinations. Total banned: {len(self.banned_combos)}")

    def select_params(self) -> Dict[str, Any]:
        """Select parameters using UCB with exploration"""
        self.t += 1

        # Періодично очищуємо погані комбінації
        if self.t % 10 == 0:
            try:
                self._check_and_ban_poor_combos()
            except Exception as e:
                log.warning(f"Auto-ban check failed: {e}")

        # Вибір параметрів
        if self.t <= 20 or random.random() < 0.3:  # exploration
            for _ in range(50):
                params = self._generate_random_params()
                combo_key = self._combo_key(params)
                if combo_key not in self.banned_combos:
                    return params
            return self._generate_random_params()
        else:  # exploitation - вибираємо найкращу за UCB
            best_ucb = -1
            best_params = None

            for combo_key, stats_dict in self.combo_stats.items():
                if combo_key in self.banned_combos:
                    continue

                N = stats_dict.get("N", 0)
                S = stats_dict.get("S", 0.0)

                if N > 0:
                    mean = S / N
                    bonus = math.sqrt(2.0 * math.log(self.t) / N)
                    ucb = mean + bonus

                    if ucb > best_ucb:
                        best_ucb = ucb
                        # Відновлюємо параметри з ключа
                        parts = combo_key.split("|")
                        if len(parts) >= 6:
                            try:
                                width, height = map(int, parts[5].split("x"))
                                best_params = {
                                    "sampler": parts[0],
                                    "scheduler": parts[1],
                                    "fps": int(parts[2]),
                                    "cfg_scale": float(parts[3]),
                                    "steps": int(parts[4]),
                                    "width": width,
                                    "height": height,
                                    "seconds": random.choice(SECONDS_OPTIONS)
                                }
                            except (ValueError, IndexError) as e:
                                log.warning(f"Failed to parse combo key {combo_key}: {e}")
                                continue

            if best_params:
                return best_params
            else:
                return self._generate_random_params()

    def update(self, params: Dict[str, Any], reward: float):
        """Update statistics for given parameters"""
        combo_key = self._combo_key(params)

        if combo_key not in self.combo_stats:
            self.combo_stats[combo_key] = {"N": 0, "S": 0.0, "scores": []}

        self.combo_stats[combo_key]["N"] += 1
        self.combo_stats[combo_key]["S"] += float(reward)
        self.combo_stats[combo_key]["scores"].append(float(reward))

        # Обмежуємо історію scores до останніх 20 спроб
        if len(self.combo_stats[combo_key]["scores"]) > 20:
            self.combo_stats[combo_key]["scores"] = self.combo_stats[combo_key]["scores"][-20:]

        self.save()


    # --- Reference-only helpers -------------------------------------------------
    def load_reference_params(self, reference_file: str = None) -> List[Dict[str, Any]]:
        """Завантаження еталонних параметрів з файлу.

        Підтримувані формати:
        - {"reference_combinations": [ {..params..} | {"params": {..}} ]}
        - {"params_list"|"combos"|"list": [ {..} | {"params":{..}} ]}
        - {"reference_videos": [ {"params": {..}} ]}
        - простий масив: [ {..} | {"params":{..}} ]
        """
        if reference_file is None:
            reference_file = os.path.join(os.path.dirname(self.state_path), "reference_params.json")

        try:
            if os.path.exists(reference_file):
                with open(reference_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                ref_items: List[Dict[str, Any]] = []

                # Helper: normalize list items to params dicts
                def _normalize_list(items: List[Any]) -> List[Dict[str, Any]]:
                    out: List[Dict[str, Any]] = []
                    for it in items:
                        if isinstance(it, dict):
                            if 'params' in it and isinstance(it['params'], dict):
                                out.append(dict(it['params']))
                            else:
                                out.append(dict(it))
                    return out

                if isinstance(data, dict):
                    for key in ("reference_combinations", "params_list", "combos", "list"):
                        if isinstance(data.get(key), list):
                            ref_items = _normalize_list(data.get(key) or [])
                            break
                    if not ref_items and isinstance(data.get("reference_videos"), list):
                        ref_items = _normalize_list(data.get("reference_videos") or [])
                elif isinstance(data, list):
                    ref_items = _normalize_list(data)

                if ref_items:
                    log.info(f"🌟 Завантажено {len(ref_items)} еталонних комбінацій")
                    return ref_items
                else:
                    log.warning("⚠️ reference_params.json порожній або має неправильну структуру")
            else:
                log.warning(f"⚠️ Файл {reference_file} не знайдено")

        except Exception as e:
            log.error(f"❌ Помилка завантаження reference_params.json: {e}")

        return []

    def select_reference_only(self, reference_combinations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вибір параметрів тільки з еталонних комбінацій (циклічно)."""
        if not reference_combinations:
            log.error("❌ Немає еталонних комбінацій для вибору!")
            return self._generate_random_params()  # Fallback

        # Рухаємо внутрішній лічильник, щоб індекс змінювався між викликами
        self.t += 1
        # Використовуємо (t-1), щоб перший виклик давав 0‑й елемент
        current_index = (self.t - 1) % len(reference_combinations)
        selected_params = dict(reference_combinations[current_index])
        log.info(f"🌟 Вибрано еталонну комбінацію {current_index + 1}/{len(reference_combinations)}")
        log.info(f"📋 Параметри: {selected_params}")
        return selected_params
