#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple

from qa.agent_namespace import agent_mod
from eva_p1.comfy_client import ComfyClient
from eva_p1.workflow import apply_t2i_params_to_workflow, apply_i2v_params_to_workflow
from eva_p1.knowledge_analyzer import KnowledgeAnalyzer
from eva_p1.prompt_generator import MegaEroticJSONPromptGenerator, EroticFullBodyPhotoPromptGenerator
from eva_p1.scenario import build_video_prompt_from_photo


def _ensure_dirs(root: str):
    for d in [
        root,
        os.path.join(root, "workflows"),
        os.path.join(root, "runs"),
        os.path.join(root, "input"),
        os.path.join(root, "output"),
        os.path.join(root, "logs"),
        os.path.join(root, "configs"),
        os.path.join(root, "state"),
    ]:
        os.makedirs(d, exist_ok=True)


def _parse_sizes(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in (spec or "").split(','):
        part = part.strip()
        if not part:
            continue
        if 'x' in part:
            try:
                w, h = part.lower().split('x', 1)
                out.append((int(w), int(h)))
            except Exception:
                pass
    return out or [(960, 540)]


def _load_reference_params(path_hint: str | None) -> List[Dict[str, Any]]:
    candidates = [p for p in [path_hint, None] if p]
    ref: List[Dict[str, Any]] = []
    for p in candidates:
        try:
            if p and os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # normalize to list of params dicts
                items: List[Dict[str, Any]] = []
                def _norm_list(li):
                    out = []
                    for it in (li or []):
                        if isinstance(it, dict):
                            out.append(it.get('params', it))
                    return out
                if isinstance(data, list):
                    items = _norm_list(data)
                elif isinstance(data, dict):
                    for key in ("reference_combinations", "params_list", "combos", "list", "reference_videos"):
                        if isinstance(data.get(key), list):
                            if key == "reference_videos":
                                items = _norm_list(data.get(key))
                            else:
                                items = _norm_list(data.get(key))
                            break
                ref = [dict(x) for x in items if isinstance(x, dict)]
                break
        except Exception:
            continue
    return ref
def _compact_prompt(text: str, max_chars: int = 600) -> str:
    """Keep prompt within reasonable size to avoid model confusion."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    # Cut on comma/space boundary if possible
    cut = t[:max_chars]
    last_comma = cut.rfind(',')
    if last_comma > 0:
        return cut[:last_comma]
    return cut


def _build_full_body_t2i_prompts(persona: Dict[str, Any]) -> Tuple[str, str]:
    # Positive: realistic full body portrait, head to toe, entire figure visible
    base = [
        "ultra realistic full-body portrait photo, head-to-toe, entire figure visible, no crop",
        "professional studio lighting, softbox, natural skin texture",
        "full-frame camera, 35mm or 50mm lens, shallow depth of field",
        "high detail, cinematic color grading, photorealistic",
    ]
    # Persona fragments
    for k in ("appearance", "hair", "ethnicity", "age_hint", "wardrobe", "pose"):
        v = persona.get(k)
        if isinstance(v, str) and v.strip():
            base.append(v.strip())
    positive = ", ".join([p for p in base if p])

    negative = ", ".join([
        "blurry, out of focus, low quality, jpeg artifacts, pixelated",
        "bad anatomy, extra limbs, missing limbs, extra fingers, fused fingers",
        "distorted face, asymmetric face, wrong proportions",
        "text, watermark, logo, signature, brand, poster",
        "cropped, out of frame, cut off feet, head cropped",
    ])
    return positive, negative


def _choose_persona(i: int) -> Dict[str, Any]:
    # Lightweight persona variety; we avoid explicit content here.
    ethnicities = ["European", "East Asian", "Latina", "Mediterranean", "Mixed race"]
    hairs = ["honey blonde", "chestnut brown", "black straight", "auburn", "platinum blonde"]
    poses = [
        "standing natural pose, balanced posture",
        "standing relaxed, slight contrapposto",
        "standing, hands gently at sides",
        "standing, one leg forward, graceful posture",
    ]
    wardrobes = [
        "minimal neutral wardrobe, clean silhouette",
        "simple elegant outfit, body-aligned fit",
        "studio wardrobe, form-consistent",
    ]
    return {
        "ethnicity": ethnicities[i % len(ethnicities)],
        "hair": hairs[i % len(hairs)],
        "age_hint": "adult 21+",
        "appearance": "natural feminine beauty, clear skin, symmetrical features",
        "pose": poses[i % len(poses)],
        "wardrobe": wardrobes[i % len(wardrobes)],
    }


def _save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_t2i2v(args):
    root = args.t2i2v_root
    _ensure_dirs(root)

    # Separate state (knowledge/bandit) under Agent_T2I2V/state
    state_dir = os.path.join(root, "state")
    os.makedirs(state_dir, exist_ok=True)

    # Prepare ComfyUI IO folders
    comfy_in = "/workspace/ComfyUI/input"
    comfy_out = "/workspace/ComfyUI/output"
    os.makedirs(comfy_in, exist_ok=True)
    os.makedirs(comfy_out, exist_ok=True)

    # Prepare run folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, "runs", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # Persist run config
    cfg = {
        "api": args.api,
        "image_workflow": args.image_workflow,
        "i2v_workflow": args.i2v_workflow,
        "image_size": [args.image_width, args.image_height],
        "i2v_sizes": args.i2v_widths,
        "i2v_fps": args.i2v_fps,
        "i2v_seconds": args.i2v_seconds,
        "iterations": args.iterations,
        "state_dir": state_dir,
    }
    _save_json(os.path.join(root, "configs", f"run_{run_id}.json"), cfg)

    # Load workflows
    if not args.image_workflow or not os.path.exists(args.image_workflow):
        raise SystemExit("Image workflow is required: --image-workflow")
    if not args.i2v_workflow or not os.path.exists(args.i2v_workflow):
        raise SystemExit("I2V workflow is required: --i2v-workflow")
    with open(args.image_workflow, 'r', encoding='utf-8') as f:
        base_t2i = json.load(f)
    with open(args.i2v_workflow, 'r', encoding='utf-8') as f:
        base_i2v = json.load(f)

    # Prepare analyzers/generators (for i2v prompt enrichment)
    knowledge = {"history": []}
    mg = MegaEroticJSONPromptGenerator(KnowledgeAnalyzer(knowledge))
    photo_gen = EroticFullBodyPhotoPromptGenerator()

    # Load reference params for video
    ref_path = args.reference_file or os.path.join(state_dir, "reference_params.json")
    ref_params = _load_reference_params(ref_path)
    if not ref_params:
        agent_mod.log.warning("⚠️ reference_params.json порожній/відсутній — відео піде з дефолтами")

    sizes = _parse_sizes(args.i2v_widths)
    client = ComfyClient(args.api)

    for i in range(int(args.iterations)):
        iter_id = f"iter_{i+1:04d}"
        iter_dir = os.path.join(run_dir, iter_id)
        os.makedirs(iter_dir, exist_ok=True)

        # Choose size (variety) and persona
        import random as _rnd
        if args.randomize_sizes:
            w, h = sizes[_rnd.randint(0, len(sizes)-1)]
        else:
            w, h = sizes[i % len(sizes)]
        persona = _choose_persona(i)
        # Build split photo prompts (two parts) and combine A+B for T2I
        if getattr(args, 'simple_prompt_test', False):
            part_a = "ultra photorealistic full-body woman, standing in a field, head-to-toe, no crop"
            part_b = "professional photography, cinematic lighting, high detail"
            t2i_neg = "blurry, low quality, jpeg artifacts, bad anatomy, extra limbs, text, watermark, cropped"
            t2i_pos = f"{part_a}, {part_b}"
        else:
            part_a, part_b, t2i_neg = photo_gen.build_photo_prompts(persona)
            t2i_pos = f"{part_a}, {part_b}"

        # T2I params (realism defaults)
        t2i_params = {
            "width": w,
            "height": h,
            "steps": 28,
            "cfg_scale": 5.0,
            "sampler": "euler",
            "scheduler": "karras",
            "seed": int(time.time() * 1000) % 2**31,
            # For Flux dual inputs (if present)
            "prompt": _compact_prompt(t2i_pos),
            "prompt_part_a": part_a,
            "prompt_part_b": part_b,
            "negative_prompt": t2i_neg,
            "prefix": f"agent_t2i2v/{os.path.basename(run_dir)}/{iter_id}/image",
        }

        # Build and queue T2I
        wf_t2i = apply_t2i_params_to_workflow(base_t2i, t2i_params)
        pid_img = client.queue(wf_t2i)
        _ = client.wait(pid_img, timeout_s=1200)

        # Resolve image path produced by SaveImage
        produced_image = None
        try:
            # best-effort: look for latest file matching prefix under comfy_out
            prefix_rel = t2i_params["prefix"].rstrip('/')
            prefix_name = os.path.basename(prefix_rel)
            # Walk output dir for files containing 'image' prefix folder
            cand_dir = os.path.join(comfy_out, os.path.dirname(prefix_rel))
            if os.path.isdir(cand_dir):
                for f in os.listdir(cand_dir):
                    if f.startswith(os.path.basename(prefix_rel)) or f.endswith('.png') or f.endswith('.jpg'):
                        produced_image = os.path.join(cand_dir, f)
                        break
        except Exception:
            pass
        if not produced_image or not os.path.exists(produced_image):
            # fallback: search any recent image
            for name in sorted(os.listdir(comfy_out), key=lambda x: os.path.getmtime(os.path.join(comfy_out, x)), reverse=True):
                p = os.path.join(comfy_out, name)
                if os.path.isfile(p) and (p.lower().endswith('.png') or p.lower().endswith('.jpg')):
                    produced_image = p
                    break
        if not produced_image:
            raise RuntimeError("Не знайдено згенероване зображення після T2I")

        # Copy to Agent_T2I2V iter folder and ComfyUI/input
        img_ext = os.path.splitext(produced_image)[1] or '.png'
        local_image = os.path.join(iter_dir, f"image{img_ext}")
        shutil.copy2(produced_image, local_image)
        input_basename = f"{os.path.basename(run_dir)}_{iter_id}_image{img_ext}"
        input_image_path = os.path.join(comfy_in, input_basename)
        shutil.copy2(local_image, input_image_path)

        # Video params from reference (do not override kombos; fill missing fields)
        i2v_params: Dict[str, Any] = {}
        if ref_params:
            src = ref_params[i % len(ref_params)]
            i2v_params.update(src)
        # Sizes: only set if missing in reference
        i2v_params.setdefault("width", w)
        i2v_params.setdefault("height", h)
        if args.randomize_fps:
            i2v_params["fps"] = int(max(args.fps_min, min(args.fps_max, _rnd.randint(args.fps_min, args.fps_max))))
        else:
            i2v_params.setdefault("fps", int(args.i2v_fps))
        i2v_params.setdefault("seconds", float(args.i2v_seconds))

        # Build i2v prompts (persona + existing erotic generator for video)
        pj = mg.generate_ultra_detailed_json_prompt()
        # Build I2V prompt strictly from photo persona + photo parts (no new random persona)
        i2v_pos = build_video_prompt_from_photo(persona, part_a, part_b, motion="slow cinematic approach, gentle arc, soft breathing, subtle hair movement")
        # Keep existing negative list for defects
        i2v_neg = mg.get_erotic_negative_prompt(pj)

        # Apply i2v workflow
        i2v_params_full = dict(i2v_params)
        i2v_params_full.update({
            "prompt": _compact_prompt(i2v_pos, max_chars=700),
            "negative_prompt": i2v_neg,
            "prefix": f"agent_t2i2v/{os.path.basename(run_dir)}/{iter_id}/video",
        })
        # Sensible video defaults if not present in reference
        i2v_params_full.setdefault("cfg_scale", 3.0)
        i2v_params_full.setdefault("steps", 8)
        i2v_params_full.setdefault("sampler", "euler")
        i2v_params_full.setdefault("scheduler", "simple")
        wf_i2v = apply_i2v_params_to_workflow(base_i2v, i2v_params_full, input_basename)
        pid_vid = client.queue(wf_i2v)
        _ = client.wait(pid_vid, timeout_s=1800)

        # Resolve video path
        produced_video = None
        try:
            cand_dir = os.path.join(comfy_out, os.path.dirname(i2v_params_full["prefix"]))
            if os.path.isdir(cand_dir):
                for f in os.listdir(cand_dir):
                    if f.startswith(os.path.basename(i2v_params_full["prefix"])) and f.lower().endswith('.mp4'):
                        produced_video = os.path.join(cand_dir, f)
                        break
        except Exception:
            pass
        if not produced_video:
            # fallback: latest mp4
            for name in sorted(os.listdir(comfy_out), key=lambda x: os.path.getmtime(os.path.join(comfy_out, x)), reverse=True):
                p = os.path.join(comfy_out, name)
                if os.path.isfile(p) and p.lower().endswith('.mp4'):
                    produced_video = p
                    break
        if not produced_video:
            raise RuntimeError("Не знайдено згенероване відео після I2V")

        local_video = os.path.join(iter_dir, "video.mp4")
        shutil.copy2(produced_video, local_video)

        # Save prompts/params/metadata
        with open(os.path.join(iter_dir, "prompt_t2i.txt"), 'w', encoding='utf-8') as f:
            f.write(t2i_params.get("prompt", ""))
        with open(os.path.join(iter_dir, "prompt_i2v.txt"), 'w', encoding='utf-8') as f:
            f.write(i2v_pos)
        _save_json(os.path.join(iter_dir, "params_t2i.json"), t2i_params)
        _save_json(os.path.join(iter_dir, "params_i2v.json"), i2v_params_full)
        _save_json(os.path.join(iter_dir, "metadata.json"), {
            "timestamp": time.time(),
            "persona": persona,
            "produced_image": produced_image,
            "produced_video": produced_video,
            "local_image": local_image,
            "local_video": local_video,
            "comfy_input_image": input_image_path,
        })

        # Update isolated knowledge under Agent_T2I2V/state
        try:
            knowledge_path = os.path.join(state_dir, "knowledge.json")
            if os.path.exists(knowledge_path):
                with open(knowledge_path, 'r', encoding='utf-8') as f:
                    K = json.load(f)
            else:
                K = {"best_score": 0.0, "best_params": {}, "history": []}
            entry = {
                "video": local_video,
                "source_image": local_image,
                "timestamp": int(time.time()),
                "params": i2v_params_full,
                "persona": persona,
                "metrics": {},
                "combo": [i2v_params_full.get('sampler'), i2v_params_full.get('scheduler')],
                "prompt": i2v_pos,
                "negative_prompt": i2v_neg,
                "photo_prompt": t2i_pos,
                "photo_negative": t2i_neg,
            }
            K.setdefault("history", []).append(entry)
            with open(knowledge_path, 'w', encoding='utf-8') as f:
                json.dump(K, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        agent_mod.log.info(f"✅ {iter_id}: image+video готові → {local_image} | {local_video}")

    agent_mod.log.info(f"🎯 Завершено двоетапний прогін: {args.iterations} ітерацій. RunDir={run_dir}")


