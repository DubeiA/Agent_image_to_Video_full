from types import SimpleNamespace

# Build a merged namespace from the smaller logical copies (no deps on eva_v4_part{1,2,3})
import eva_env_base as _env  # provides log and env constants
import eva_p1.analysis_config as _p1a
import eva_p1.prompt_generator as _p1b
import eva_p1.comfy_client as _p1c_client
import eva_p1.video_analyzer as _p1c_an
import eva_p1.multi_bandit as _p1c_bandit
import eva_p1.workflow as _p1w
import eva_p1.agent_base as _p1agent
import eva_p1.openrouter_analyzer as _p1gpt
import eva_p2.merged_agent as _p2a
import eva_p2.cli_patch as _p2b
import eva_p3.logger as _p3a
try:
    import eva_p3.enhanced_analyzer as _p3b  # optional legacy module
except Exception:
    _p3b = None
import eva_p3.video_processor as _p3c
try:
    import eva_p3.training as _p3d  # optional
except Exception:
    _p3d = None

agent_mod = SimpleNamespace()

# Prefer logger from part3 if available, else from part1
setattr(agent_mod, 'EnhancedLogger', getattr(_p3a, 'EnhancedLogger', None))
# Video processing & improved analyzer
setattr(agent_mod, 'VideoProcessor', getattr(_p3c, 'VideoProcessor', None))
# Bandit and simple analyzer clients
setattr(agent_mod, 'MultiDimensionalBandit', getattr(_p1c_bandit, 'MultiDimensionalBandit', None))
setattr(agent_mod, 'ComfyClient', getattr(_p1c_client, 'ComfyClient', None))
setattr(agent_mod, 'VideoAnalyzer', getattr(_p1c_an, 'VideoAnalyzer', None))
# OpenRouter analyzer (GPT)
setattr(agent_mod, 'OpenRouterAnalyzer', getattr(_p1gpt, 'OpenRouterAnalyzer', None))
# Main merged Agent
setattr(agent_mod, 'EnhancedVideoAgentV4Merged', getattr(_p2a, 'EnhancedVideoAgentV4Merged', None))
setattr(agent_mod, 'EnhancedVideoAgentV4', getattr(_p1agent, 'EnhancedVideoAgentV4', None))
# Module-level logger if present
setattr(agent_mod, 'log', getattr(_env, 'log', None))
setattr(agent_mod, 'validate_workflow_nodes', getattr(_p1w, 'validate_workflow_nodes', None))
setattr(agent_mod, 'apply_enhanced_params_to_workflow', getattr(_p1w, 'apply_enhanced_params_to_workflow', None))


