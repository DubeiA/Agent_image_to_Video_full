from typing import Dict, Any, List
from eva_p1.scenario import generate_mega_erotic_scenario
from eva_p1.knowledge_analyzer import KnowledgeAnalyzer


class MegaEroticJSONPromptGenerator:
    """Generates mega-expanded erotic JSON prompts based on knowledge analysis"""
    
    def __init__(self, knowledge_analyzer):
        self.analyzer = knowledge_analyzer
        self.load_advanced_templates()
        
    def load_advanced_templates(self):
        """Load advanced prompt component templates"""
        self.cinematography_templates = {
            "professional_erotic": {
                "camera_type": "Professional cinema camera with full-frame sensor optimized for intimate scenes",
                "lens_options": [
                    "85mm prime lens with shallow depth of field for intimate focus",
                    "50mm f/1.4 for natural perspective with sensual bokeh", 
                    "135mm for compression and dreamy background blur",
                    "35mm f/1.8 for wider intimate environment capture"
                ],
                "movements": [
                    {"type": "slow_sensual_dolly_in", "quality": "buttery_smooth_seductive"},
                    {"type": "gentle_arc_movement", "smoothness": "gimbal_stabilized_intimate"},
                    {"type": "subtle_zoom", "speed": "imperceptible_slow_focus_on_curves"},
                    {"type": "smooth_pan", "style": "following_body_contours"}
                ]
            }
        }
        
        self.lighting_templates = {
            "sensual_studio": {
                "primary": "Professional 3-point lighting with large softboxes for flattering skin tones",
                "fill": "Bounce reflectors with warm diffused fill light enhancing curves",
                "accent": [
                    "Neon practical lights in pink, purple, blue creating mood",
                    "LED strips for rim lighting accentuating body contours",
                    "Colored gels for dramatic intimate highlights"
                ],
                "ratios": "3:1 key to fill ratio for dramatic depth and sensual shadows"
            }
        }
    
    def generate_ultra_detailed_json_prompt(self) -> Dict[str, Any]:
        """Generate ultra-detailed JSON prompt with mega erotic scenarios"""
        scenario = generate_mega_erotic_scenario()
        insights = self.analyzer.generate_improvement_insights()
        
        prompt_json = {
            "video_production": {
                "title": self._generate_erotic_scene_title(scenario),
                "duration": "8-10 seconds of pure sensual cinematography",
                "aspect_ratio": "16:9 cinematic format",
                "resolution": "4K Ultra HD with crystal clarity",
                "production_style": "Ultra-luxury erotic cinematography with professional standards",
                "generation_strategy": self._get_erotic_generation_strategy(insights),
                
                "subject_details": self._generate_detailed_subject(scenario),
                "environment": self._design_luxury_environment(scenario),
                "quality_enhancements": self._get_erotic_quality_enhancements(insights),
                "defect_prevention": self._get_comprehensive_defect_prevention(insights)
            }
        }
        
        return prompt_json
    
    def _generate_detailed_subject(self, scenario: Dict) -> Dict:
        """Generate ultra-detailed subject with mega erotic elements"""
        return {
            "character_description": {
                "appearance_details": scenario["appearance"],
                "build_description": scenario["body_type"]
            },
            
            "wardrobe_and_styling": {
                "main_outfit": {
                    "description": scenario["clothing"],
                    "fit_and_style": "Perfectly fitted to emphasize natural curves and assets"
                }
            },
            
            "quality_focus": {
                "anatomy_perfection": "Flawless human proportions and realistic anatomy",
                "facial_beauty": "Symmetric gorgeous facial features with perfect skin",
                "hand_and_finger_accuracy": "Properly formed hands with correct finger count",
                "body_consistency": "Consistent body proportions and physics throughout video"
            }
        }
    
    def _design_luxury_environment(self, scenario: Dict) -> Dict:
        """Design ultra-luxury environment from scenario"""
        return {
            "location_details": {
                "primary_setting": scenario["location"],
                "activity_description": scenario["activity"],
                "atmosphere": "Luxurious and intimate with professional ambiance"
            }
        }
    
    def _generate_erotic_scene_title(self, scenario: Dict) -> str:
        """Generate compelling erotic scene title"""
        appearance_parts = scenario["appearance"].split()
        age = next((part.split("-")[0] for part in appearance_parts if "year-old" in part), "25")
        
        location_type = scenario["location_key"].replace("_", " ").title()
        
        body_emphasis = ""
        if "enormous" in scenario["body_type"] or "gigantic" in scenario["body_type"]:
            if "breast" in scenario["body_type"]:
                body_emphasis = "Busty "
            elif "ass" in scenario["body_type"] or "butt" in scenario["body_type"]:
                body_emphasis = "Curvy "
            else:
                body_emphasis = "Voluptuous "
        
        return f"Sensual {age}-Year-Old {body_emphasis}Beauty in {location_type}"
    
    def _get_erotic_quality_enhancements(self, insights: Dict) -> List[str]:
        """Get quality enhancements for erotic content"""
        return [
            "Ultra-high definition skin rendering with natural texture",
            "Professional cinematography with intimate framing",
            "Perfect anatomical accuracy with enhanced feminine appeal"
        ]
    
    def _get_comprehensive_defect_prevention(self, insights: Dict) -> Dict:
        """Get comprehensive defect prevention for erotic content"""
        return {
            "positive_reinforcement": [
                "Perfect human anatomy with realistic proportions",
                "Correct number of limbs and fingers",
                "Beautiful symmetric facial features",
                "High-quality rendering without artifacts"
            ],
            "negative_prompts": [
                "deformed, malformed, distorted",
                "extra arms, extra legs, extra fingers",
                "distorted face, asymmetric face",
                "blurry, low quality, pixelated"
            ]
        }
    
    def _get_erotic_generation_strategy(self, insights: Dict) -> str:
        """Get generation strategy for erotic content"""
        return "Ultra-luxury erotic cinematography focused on quality and realism"
    
    def convert_to_erotic_text_prompt(self, json_prompt: Dict) -> str:
        """Convert JSON prompt to enhanced erotic text for ComfyUI"""
        video_data = json_prompt["video_production"]
        
        # Generate mega erotic scenario for text conversion
        mega_scenario = generate_mega_erotic_scenario()
        
        # Build comprehensive text prompt
        parts = []
        
        # Technical quality opening
        parts.append("PROFESSIONAL EROTIC VIDEO: Ultra-high definition 4K cinematic production")
        parts.append("Professional cinema camera with 85mm prime lens")
        
        # Subject details from scenario
        parts.append(f"FEATURED MODEL: {mega_scenario['appearance']}")
        parts.append(mega_scenario['body_type'])
        
        # Wardrobe
        parts.append(f"WARDROBE: {mega_scenario['clothing']}")
        
        # Environment and activity
        parts.append(f"SCENE: {mega_scenario['activity']} in {mega_scenario['location']}")
        
        # Quality enhancements
        quality_enhancements = video_data.get("quality_enhancements", [])
        if quality_enhancements:
            parts.extend(quality_enhancements[:3])
        
        # Defect prevention positive prompts
        prevention = video_data.get("defect_prevention", {})
        if "positive_reinforcement" in prevention:
            parts.extend(prevention["positive_reinforcement"][:3])
        
        # Final production notes
        parts.append("Natural authentic feminine movement with professional video standards")
        parts.append("Luxury aesthetic with sophisticated erotic cinematography")
        
        return ", ".join([p for p in parts if p])
    
    def get_erotic_negative_prompt(self, json_prompt: Dict) -> str:
        """Generate comprehensive negative prompt for erotic content"""
        prevention = json_prompt["video_production"].get("defect_prevention", {})
        
        negative_parts = [
            # Chinese negative terms
            "色调艳丽，过曝，静态，细节模糊不清，最差质量，低质量，丑陋的，残缺的",
            "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体",
            
            # English negative terms
            "deformed, malformed, extra arms, extra legs, distorted face, anatomical errors",
            "blurry, low quality, worst quality, jpeg artifacts, duplicate, morbid, mutilated",
            "bad anatomy, bad proportions, extra fingers, fused fingers, artificial, plastic, fake"
        ]
        
        # Add specific negative prompts from prevention
        if "negative_prompts" in prevention:
            negative_parts.extend(prevention["negative_prompts"])
        
        return ", ".join(negative_parts)


# --- New: Full-body nude photo prompt generator (two-part structure) ---

class EroticFullBodyPhotoPromptGenerator:
    """Builds a highly realistic full-body erotic (non-nude) photo prompt (adult 21+) in two parts.

    Part A (persona/technical): camera/lens/lighting/composition ensuring head-to-toe visibility.
    Part B (erotic/styling): lingerie/sheer fabrics, sensual aesthetics, posing (no explicit acts, not nude).

    Returns (part_a, part_b, negative_prompt).
    """

    def __init__(self):
        # Static technical baselines for realism
        self.technical = [
            "ultra realistic photographic quality",
            "full-body, head-to-toe, entire figure visible, no crop",
            "balanced centered composition, feet fully visible",
            "full-frame camera, 35mm or 50mm lens",
            "studio softbox lighting with natural skin rendering",
            "subtle cinematic color grading, high micro-contrast",
        ]

        # Erotic styling (non-nude, tasteful lingerie aesthetics; strictly adult)
        self.erotic = [
            "tasteful erotic lingerie (adult 21+), refined feminine curves",
            "luxury lace lingerie or sheer silk robe, implied coverage",
            "authentic skin texture, delicate highlights and shadows",
            "elegant standing pose, relaxed posture, graceful silhouette",
            "clean minimal background",
        ]

        # Negative constraints to prevent defects/illegal content
        self.negative = [
            "minor, child, teen, young-looking, underage",
            "blurry, low quality, jpeg artifacts, pixelated",
            "bad anatomy, extra limbs, missing limbs, extra fingers, fused fingers",
            "distorted face, asymmetrical face, wrong proportions",
            "text, watermark, logo, signature, brand, poster",
            "cropped, out of frame, head cut, feet cut",
            "fully nude, explicit nudity, exposed nipples, genitalia",
            "NSFW acts, explicit sexual acts, pornographic acts",
        ]

    def build_photo_prompts(self, persona: Dict[str, Any]) -> (str, str, str):
        # Persona fragments (adult emphasis and variety hooks)
        persona_parts: List[str] = []
        appearance = persona.get("appearance")
        if isinstance(appearance, str) and appearance.strip():
            persona_parts.append(appearance.strip())
        if isinstance(persona.get("hair"), str) and persona["hair"].strip():
            persona_parts.append(persona["hair"].strip())
        if isinstance(persona.get("ethnicity"), str) and persona["ethnicity"].strip():
            persona_parts.append(persona["ethnicity"].strip())
        persona_parts.append("adult 21+")

        part_a = ", ".join(persona_parts + self.technical)
        part_b = ", ".join(self.erotic)
        negative = ", ".join(self.negative)
        return part_a, part_b, negative

