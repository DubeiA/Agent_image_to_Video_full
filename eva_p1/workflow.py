# Copied from eva_p1_workflow_and_agent.py
import json
from typing import Dict, Any
from eva_env_base import log

def validate_workflow_nodes(workflow: Dict[str, Any]) -> bool:
    """Validate that workflow has required nodes"""
    required_nodes = ["72", "74", "78", "80", "81", "88", "89"]
    missing_nodes = []

    for node_id in required_nodes:
        if node_id not in workflow:
            missing_nodes.append(node_id)

    if missing_nodes:
        log.error(f"Missing workflow nodes: {missing_nodes}")
        return False

    log.info("✅ Workflow validation passed")
    return True


def apply_enhanced_params_to_workflow(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply enhanced parameters to ComfyUI workflow"""
    wf = json.loads(json.dumps(base))

    # Apply prompt
    if "prompt" in params and "89" in wf:
        wf["89"]["inputs"]["text"] = params["prompt"]

    # Apply negative prompt
    if "negative_prompt" in params and "72" in wf:
        wf["72"]["inputs"]["text"] = params["negative_prompt"]

    # Apply video settings
    fps = int(params.get("fps", 20))
    seconds = float(params.get("seconds", 5.0))

    if "88" in wf:
        wf["88"]["inputs"]["fps"] = fps

    if "74" in wf:
        wf["74"]["inputs"]["length"] = int(max(1, round(fps * seconds)))
        if "width" in params and "height" in params:
            wf["74"]["inputs"]["width"] = int(params["width"])
            wf["74"]["inputs"]["height"] = int(params["height"])

    # Apply sampler settings
    if params.get("sampler") or params.get("scheduler"):
        sampler = params["sampler"]
        scheduler = params["scheduler"]

        # High resolution pass
        if "81" in wf:
            wf["81"]["inputs"]["sampler_name"] = sampler
            wf["81"]["inputs"]["scheduler"] = scheduler
            if "steps" in params:
                wf["81"]["inputs"]["steps"] = int(params["steps"])
            if "cfg_scale" in params:
                wf["81"]["inputs"]["cfg"] = float(params["cfg_scale"])

        # Low resolution pass
        if "78" in wf:
            wf["78"]["inputs"]["sampler_name"] = sampler
            wf["78"]["inputs"]["scheduler"] = scheduler
            if "steps" in params:
                wf["78"]["inputs"]["steps"] = int(params["steps"])  
            if "cfg_scale" in params:
                wf["78"]["inputs"]["cfg"] = float(params["cfg_scale"])

    # Apply seeds
    if "seed_high" in params and "81" in wf:
        wf["81"]["inputs"]["noise_seed"] = int(params["seed_high"])
    if "seed_low" in params and "78" in wf:
        wf["78"]["inputs"]["noise_seed"] = int(params["seed_low"])

    # Apply output prefix to all output nodes we know about
    prefix = params.get("prefix")
    if prefix:
        for node_id in ("80", "90", "100", "110"):
            if node_id in wf and isinstance(wf[node_id].get("inputs"), dict) and "filename_prefix" in wf[node_id]["inputs"]:
                wf[node_id]["inputs"]["filename_prefix"] = prefix

    return wf


def apply_t2i_params_to_workflow(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply parameters to a generic T2I workflow (FLUX/SDXL-like).

    Expected node conventions (best-effort):
    - Positive text node key: first node with class_type CLIPTextEncode/FluxTextEncode (pos)
    - Negative text node key: second node with same class (neg)
    - EmptyLatentImage-like: node with inputs width/height
    - KSampler-like: node with inputs steps/cfg/sampler_name/scheduler/seed
    - SaveImage: node with inputs filename_prefix
    """
    import json as _json
    wf = _json.loads(_json.dumps(base))

    # Find text nodes (Flux/CLIP/CLIPTextEncodeFlux)
    pos_key, neg_key, latent_key, ksampler_key, save_key = None, None, None, None, None
    for k, v in wf.items():
        cls = (v or {}).get('class_type') or ''
        title = ((v or {}).get('_meta') or {}).get('title', '')
        if cls in ("CLIPTextEncode", "FluxTextEncode", "CLIPTextEncodeFlux"):
            if ('negative' in title.lower()) and (neg_key is None):
                neg_key = k
            elif pos_key is None:
                pos_key = k
        if isinstance((v or {}).get('inputs'), dict):
            ins = v['inputs']
            if all(x in ins for x in ("width", "height")):
                latent_key = latent_key or k
        if isinstance((v or {}).get('inputs'), dict) and any(x in v['inputs'] for x in ("sampler_name", "steps", "cfg")):
            if v.get('class_type', '').lower().startswith('ksampler') or 'KSampler' in v.get('class_type', ''):
                ksampler_key = ksampler_key or k
        if v.get('class_type') == 'SaveImage' and isinstance(v.get('inputs'), dict):
            save_key = save_key or k

    if pos_key and isinstance(wf[pos_key].get('inputs'), dict):
        ins = wf[pos_key]['inputs']
        # Flux dual-field support
        if 'text_g' in ins or 'text_l' in ins:
            if 'text_g' in ins:
                ins['text_g'] = params.get('prompt_part_a', ins['text_g'])
            if 'text_l' in ins:
                ins['text_l'] = params.get('prompt_part_b', ins['text_l'])
        # Fallback single field
        if 'text' in ins:
            ins['text'] = params.get('prompt', ins['text'])
    if neg_key and isinstance(wf[neg_key].get('inputs'), dict):
        insn = wf[neg_key]['inputs']
        if 'text_g' in insn or 'text_l' in insn:
            # Put negatives split if desired; for now entire negative into both to maximize effect
            neg = params.get('negative_prompt')
            if 'text_g' in insn:
                insn['text_g'] = neg or insn['text_g']
            if 'text_l' in insn:
                insn['text_l'] = neg or insn['text_l']
        if 'text' in insn:
            insn['text'] = params.get('negative_prompt', insn['text'])
    if neg_key and isinstance(wf[neg_key].get('inputs'), dict) and 'text' in wf[neg_key]['inputs']:
        wf[neg_key]['inputs']['text'] = params.get('negative_prompt', wf[neg_key]['inputs']['text'])
    if latent_key and isinstance(wf[latent_key].get('inputs'), dict):
        wf[latent_key]['inputs']['width'] = int(params.get('width', wf[latent_key]['inputs'].get('width', 960)))
        wf[latent_key]['inputs']['height'] = int(params.get('height', wf[latent_key]['inputs'].get('height', 540)))
    if ksampler_key and isinstance(wf[ksampler_key].get('inputs'), dict):
        ins = wf[ksampler_key]['inputs']
        # Steps variants
        for key in ('steps', 'num_steps'):
            if key in ins:
                ins[key] = int(params.get('steps', ins[key]))
        # CFG/guidance variants
        if 'cfg' in ins:
            ins['cfg'] = float(params.get('cfg_scale', ins['cfg']))
        if 'guidance' in ins:
            ins['guidance'] = float(params.get('cfg_scale', ins['guidance']))
        # Sampler variants
        if 'sampler_name' in ins:
            ins['sampler_name'] = params.get('sampler', ins['sampler_name'])
        if 'sampler' in ins:
            ins['sampler'] = params.get('sampler', ins['sampler'])
        if 'scheduler' in ins:
            ins['scheduler'] = params.get('scheduler', ins['scheduler'])
        # Seed variants
        if 'noise_seed' in ins:
            ins['noise_seed'] = int(params.get('seed', ins['noise_seed']))
        if 'seed' in ins:
            ins['seed'] = int(params.get('seed', ins['seed']))
    if save_key and isinstance(wf[save_key].get('inputs'), dict) and 'filename_prefix' in wf[save_key]['inputs']:
        wf[save_key]['inputs']['filename_prefix'] = params.get('prefix', wf[save_key]['inputs']['filename_prefix'])
    return wf


def apply_i2v_params_to_workflow(base: Dict[str, Any], params: Dict[str, Any], input_image_name: str) -> Dict[str, Any]:
    """Apply parameters to WAN i2v workflow (video_wan2_2_14B_i2v.json)."""
    import json as _json
    wf = _json.loads(_json.dumps(base))

    # Prompts
    if "93" in wf and isinstance(wf["93"].get('inputs'), dict):
        wf["93"]["inputs"]["text"] = params.get("prompt", wf["93"]["inputs"].get("text"))
    if "89" in wf and isinstance(wf["89"].get('inputs'), dict):
        wf["89"]["inputs"]["text"] = params.get("negative_prompt", wf["89"]["inputs"].get("text"))

    # Sizes and video length
    w = int(params.get("width", 960))
    h = int(params.get("height", 540))
    fps = int(params.get("fps", 20))
    seconds = float(params.get("seconds", 6.0))
    length = max(1, int(round(fps * seconds)))

    if "98" in wf and isinstance(wf["98"].get('inputs'), dict):
        wf["98"]["inputs"]["width"] = w
        wf["98"]["inputs"]["height"] = h
        wf["98"]["inputs"]["length"] = length
    if "94" in wf and isinstance(wf["94"].get('inputs'), dict):
        wf["94"]["inputs"]["fps"] = fps

    # Sampler/cfg/steps (both KSampler nodes 85/86 might exist)
    for node_id in ("85", "86"):
        if node_id in wf and isinstance(wf[node_id].get('inputs'), dict):
            ins = wf[node_id]['inputs']
            if 'sampler_name' in ins:
                ins['sampler_name'] = params.get('sampler', ins['sampler_name'])
            if 'scheduler' in ins:
                ins['scheduler'] = params.get('scheduler', ins['scheduler'])
            if 'steps' in ins:
                ins['steps'] = int(params.get('steps', ins['steps']))
            if 'cfg' in ins:
                ins['cfg'] = float(params.get('cfg_scale', ins['cfg']))

    # Input image
    if "97" in wf and isinstance(wf["97"].get('inputs'), dict):
        wf["97"]["inputs"]["image"] = input_image_name

    # Save video prefix
    if "108" in wf and isinstance(wf["108"].get('inputs'), dict):
        if 'filename_prefix' in wf["108"]["inputs"]:
            wf["108"]["inputs"]["filename_prefix"] = params.get('prefix', wf["108"]["inputs"]["filename_prefix"])
    return wf

