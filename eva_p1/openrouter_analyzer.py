# Copied from eva_p1_config_gpt.py (requires openai, log, GPT_AVAILABLE, re, json)
from typing import Dict, Any
import re, json
from eva_env_base import log, GPT_AVAILABLE
try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

class OpenRouterAnalyzer:
    """Аналізатор ручних оцінок через OpenRouter GPT"""
    
    def __init__(self, api_key: str):
        if not GPT_AVAILABLE or openai is None:
            raise ImportError("OpenAI library required for GPT integration")

        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        log.info("🤖 OpenRouter GPT analyzer initialized successfully")
    
    def analyze_manual_rating(self, video_name: str, rating_data: Dict) -> Dict[str, Any]:
        """Глибокий аналіз ручної оцінки через GPT"""
        
        rating = rating_data.get("rating", {})
        
        # Формуємо промпт для аналізу
        analysis_prompt = f"""You are a video generation quality expert. Analyze this manual video rating and extract actionable insights for improving video generation parameters.

VIDEO: {video_name}
RATINGS (1-10 scale):
- Overall Quality: {rating.get('overall_quality', 'N/A')}/10
- Visual Quality: {rating.get('visual_quality', 'N/A')}/10  
- Motion Quality: {rating.get('motion_quality', 'N/A')}/10
- Prompt Adherence: {rating.get('prompt_adherence', 'N/A')}/10
- Creativity: {rating.get('creativity', 'N/A')}/10
- Technical Quality: {rating.get('technical_quality', 'N/A')}/10

DETECTED DEFECTS:
- Anatomy Issues: {rating.get('anatomy_issues', False)}
- Face Distortion: {rating.get('face_distortion', False)}
- Temporal Inconsistency: {rating.get('temporal_inconsistency', False)}
- Artifacts: {rating.get('artifacts', False)}
- Lighting Issues: {rating.get('lighting_issues', False)}

USER COMMENTS: "{rating.get('comments', 'No comments')}"
MARKED AS REFERENCE: {rating.get('is_reference', False)}

Based on this analysis, provide recommendations in JSON format:
{
    "quality_score": 0.75,
    "main_problems": ["lighting_issues", "face_distortion"],
    "parameter_adjustments": {
        "sampler_preference": "prefer_stable|avoid_current|no_preference",
        "scheduler_preference": "prefer_simple|prefer_normal|prefer_karras|no_preference", 
        "fps_adjustment": "increase|decrease|maintain",
        "steps_adjustment": "increase|decrease|maintain",
        "cfg_adjustment": "increase|decrease|maintain"
    },
    "prompt_improvements": ["add better lighting description", "focus on face quality"],
    "technical_fixes": ["use lower FPS for stability", "increase steps for detail"],
    "is_reference_worthy": true,
    "learning_priority": "high|medium|low"
}

Focus on actionable technical recommendations based on the ratings and comments."""

        try:
            log.info(f"🤖 Analyzing {video_name} with GPT...")
            
            response = self.client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            log.info(f"🤖 GPT response received for {video_name} ({len(analysis_text)} chars)")
            
            # Витягуємо JSON з відповіді
            json_match = re.search(r'\{[\s\S]*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                log.info(f"✅ GPT Analysis for {video_name}: score={analysis.get('quality_score', 0):.3f}, problems={analysis.get('main_problems', [])}")
                return analysis
            else:
                log.warning(f"❌ Failed to parse GPT JSON for {video_name}")
                return self._fallback_analysis(rating_data)
                
        except Exception as e:
            log.error(f"❌ OpenRouter analysis failed for {video_name}: {e}")
            return self._fallback_analysis(rating_data)
    
    def _fallback_analysis(self, rating_data: Dict) -> Dict[str, Any]:
        """Fallback аналіз якщо GPT недоступний"""
        rating = rating_data.get("rating", {})
        overall = rating.get("overall_quality", 5)
        
        # Базовий аналіз на основі оцінок
        problems = []
        if rating.get("anatomy_issues"): problems.append("anatomy_issues")
        if rating.get("face_distortion"): problems.append("face_distortion")
        if rating.get("artifacts"): problems.append("artifacts")
        if rating.get("lighting_issues"): problems.append("lighting_issues")
        
        return {
            "quality_score": max(0.0, min(1.0, (overall - 1) / 9)),
            "main_problems": problems or ["unknown"],
            "parameter_adjustments": {
                "sampler_preference": "no_preference",
                "scheduler_preference": "no_preference",
                "fps_adjustment": "maintain",
                "steps_adjustment": "maintain",
                "cfg_adjustment": "maintain"
            },
            "prompt_improvements": [],
            "technical_fixes": [],
            "is_reference_worthy": overall >= 8,
            "learning_priority": "high" if overall <= 3 or overall >= 8 else "medium"
        }

