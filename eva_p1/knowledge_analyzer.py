# Copied from eva_p1_prompts.py
from typing import Dict, Any
import numpy as np

class KnowledgeAnalyzer:
    """Analyzes knowledge database for intelligent prompt generation"""
    
    def __init__(self, knowledge_data: Dict[str, Any]):
        self.knowledge = knowledge_data
        
    def generate_improvement_insights(self) -> Dict[str, Any]:
        """Generate insights for improvement based on knowledge history"""
        history = self.knowledge.get("history", [])
        if not history:
            return {"defect_prevention": [], "quality_enhancements": []}
            
        # Analyze common issues
        low_scores = [entry for entry in history if entry.get("score", 0) < 0.5]
        high_scores = [entry for entry in history if entry.get("score", 0) > 0.8]
        
        insights = {
            "defect_prevention": [],
            "quality_enhancements": []
        }
        
        # Analyze metrics patterns
        if low_scores:
            avg_blur = np.mean([entry.get("metrics", {}).get("blur", 0) for entry in low_scores])
            avg_exposure = np.mean([entry.get("metrics", {}).get("exposure", 0) for entry in low_scores])
            
            if avg_blur < 0.3:
                insights["defect_prevention"].append({
                    "defect": "blur_issues",
                    "avg_severity": avg_blur * 10,
                    "prevention_tips": ["Focus on sharp details", "Use proper focus techniques"]
                })
                
            if avg_exposure < 0.4:
                insights["defect_prevention"].append({
                    "defect": "lighting_issues", 
                    "avg_severity": avg_exposure * 10,
                    "prevention_tips": ["Improve lighting setup", "Avoid overexposure"]
                })
        
        if high_scores:
            insights["quality_enhancements"] = [
                {"recommendation": "Maintain high cinematography standards"},
                {"recommendation": "Focus on professional lighting"}
            ]
            
        return insights
    
    def analyze_successful_patterns(self) -> Dict[str, Any]:
        """Analyze patterns from successful generations"""
        history = self.knowledge.get("history", [])
        high_quality_videos = [
            {
                "score": entry.get("score", 0),
                "data": {
                    "basic_quality": {
                        "lighting_exposure": entry.get("metrics", {}).get("exposure", 0) * 10,
                        "blur_sharpness": entry.get("metrics", {}).get("blur", 0) * 10
                    },
                    "content_accuracy": {
                        "character_consistency": entry.get("score", 0) * 10,
                        "clothing_accuracy": entry.get("score", 0) * 10
                    }
                }
            }
            for entry in history if entry.get("score", 0) > 0.7
        ]
        
        return {"high_quality_videos": high_quality_videos}
    
    def suggest_next_experiment(self) -> Dict[str, str]:
        """Suggest next experimental focus"""
        history = self.knowledge.get("history", [])
        if not history:
            return {"focus_area": "quality", "reasoning": "Initial quality establishment"}
            
        recent_avg = np.mean([entry.get("score", 0) for entry in history[-5:]])
        
        if recent_avg < 0.6:
            return {"focus_area": "basic_quality", "reasoning": "Improve fundamental quality metrics"}
        elif recent_avg < 0.8:
            return {"focus_area": "advanced_features", "reasoning": "Enhance cinematography and lighting"}
        else:
            return {"focus_area": "creative_variation", "reasoning": "Explore creative possibilities"}

