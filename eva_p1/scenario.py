# Copied from eva_p1_prompts.py
import random

def generate_mega_erotic_scenario():
    """Генерує мега-розширені еротичні сценарії з великим різноманіттям"""

    # 50+ ЛОКАЦІЙ (без каміння/вогню/їжі)
    locations = {
        "luxury_bedroom": {
            "base": "luxurious master bedroom with silk sheets and ambient lighting",
            "variations": ["penthouse bedroom", "hotel presidential suite", "mansion master bedroom", "yacht luxury cabin", "castle royal chamber", "private villa bedroom", "mountain chalet bedroom"],
            "activities": [
                "posing seductively on silk sheets with golden hour lighting",
                "stretching sensually by floor-to-ceiling windows",
                "lying gracefully across satin pillows with candles around",
                "sitting on bed edge in revealing lingerie with soft shadows",
                "kneeling on bed with arms above head emphasizing curves",
                "arching back provocatively on luxury mattress",
                "rolling seductively across silk bedding",
                "sitting cross-legged on velvet cushions with mysterious smile",
                "leaning back against ornate headboard with elegant posture",
                "gracefully adjusting silk stockings while seated on bed edge"
            ]
        },
        "spa_bathroom": {
            "base": "elegant spa bathroom with marble surfaces and warm lighting", 
            "variations": ["marble bathroom", "infinity spa", "luxury hotel bathroom", "private wellness center", "Roman bath house", "modern zen bathroom", "vintage clawfoot bathroom"],
            "activities": [
                "stepping out of steam shower with water droplets glistening",
                "sitting by marble bathtub with rose petals and candles",
                "applying luxurious lotion in front of gold-framed mirror",
                "relaxing in spa atmosphere with soft towels",
                "posing by marble countertop with professional lighting",
                "stretching sensually in marble shower with steam",
                "leaning seductively against bathroom mirror",
                "gracefully testing water temperature with elegant toe dip",
                "sitting on marble bench adjusting silk robe",
                "posing with fluffy towel draped artistically around curves"
            ]
        },
        "modern_penthouse": {
            "base": "contemporary penthouse living space with city skyline view",
            "variations": ["minimalist penthouse", "glass penthouse", "luxury high-rise", "modern loft", "sky villa", "urban apartment", "designer studio"],
            "activities": [
                "lounging seductively on Italian leather furniture",
                "posing against floor-to-ceiling windows with city lights",
                "sitting cross-legged on designer carpet with natural grace",
                "stretching on modern furniture with architectural lighting",
                "relaxing by contemporary furniture with sophisticated ambiance",
                "dancing provocatively by panoramic windows",
                "reclining on luxury sofa with legs elegantly positioned",
                "standing confidently by modern art installation",
                "sitting elegantly on designer bar stool",
                "posing playfully on geometric furniture pieces"
            ]
        },
        "executive_office": {
            "base": "high-end executive office with mahogany furniture and city view",
            "variations": ["CEO office", "law firm office", "penthouse office", "corporate boardroom", "bank president office", "media company office", "fashion agency office"],
            "activities": [
                "sitting provocatively on executive mahogany desk",
                "leaning against floor-to-ceiling bookshelf seductively",
                "posing in leather executive chair with confidence",
                "standing by panoramic windows in business attire",
                "taking confident break from work in revealing outfit",
                "stretching sensually during office break",
                "posing with documents scattered suggestively",
                "sitting confidently on conference table edge",
                "leaning against office door frame with authority",
                "adjusting business attire while standing by window"
            ]
        },
        "fashion_studio": {
            "base": "professional photography studio with fashion lighting setup",
            "variations": ["modeling studio", "portrait studio", "commercial studio", "artistic loft studio", "fashion showroom", "designer atelier"],
            "activities": [
                "posing under professional studio lighting with confidence",
                "adjusting pose on seamless white backdrop",
                "sitting gracefully on photography cube prop",
                "stretching elegantly against studio wall",
                "dancing freely in open studio space",
                "posing with fabric draping artistically",
                "sitting on director's chair with legs crossed elegantly",
                "leaning against studio equipment with casual confidence",
                "adjusting hair while standing by makeup station",
                "posing with vintage photography props"
            ]
        },
        "luxury_yacht": {
            "base": "private luxury yacht deck with ocean views and elegant furnishing",
            "variations": ["mega yacht deck", "sailing yacht cabin", "motor yacht lounge", "yacht sun deck", "floating villa", "cruise ship suite"],
            "activities": [
                "lounging on yacht deck chairs with ocean breeze",
                "posing by yacht railing with sunset backdrop",
                "sitting on deck furniture with legs elegantly positioned",
                "stretching on yacht deck with maritime elegance",
                "relaxing in yacht cabin with luxury ambiance",
                "dancing on deck under starlight",
                "sitting on yacht edge with feet elegantly dangling",
                "posing with nautical elements in background",
                "adjusting swimwear while enjoying ocean view",
                "leaning against yacht mast with confident smile"
            ]
        },
        "art_gallery": {
            "base": "sophisticated modern art gallery with white walls and sculptures",
            "variations": ["contemporary gallery", "classic museum", "private collection", "sculpture garden indoor", "artistic warehouse", "cultural center"],
            "activities": [
                "posing thoughtfully next to abstract sculptures",
                "sitting elegantly on gallery bench contemplating art",
                "leaning against white gallery wall with artistic confidence",
                "walking gracefully through gallery corridors",
                "posing with modern art pieces as backdrop",
                "sitting cross-legged on gallery floor with artistic expression",
                "stretching elegantly in spacious gallery room",
                "adjusting pose while observing artwork",
                "standing confidently by large artistic installation",
                "sitting on sculptural bench with legs elegantly crossed"
            ]
        },
        "luxury_library": {
            "base": "elegant private library with floor-to-ceiling bookshelves and reading areas",
            "variations": ["mansion library", "university rare books room", "private study", "literary salon", "book collector's room", "academic reading room"],
            "activities": [
                "sitting elegantly in leather reading chair with book",
                "reaching gracefully for books on high shelves",
                "posing thoughtfully by antique reading desk",
                "leaning against bookshelf ladder with intellectual charm",
                "sitting cross-legged on library carpet reading",
                "stretching elegantly in quiet reading nook",
                "adjusting reading glasses while holding classic book",
                "posing confidently by rare book collection",
                "sitting on library steps with book in hand",
                "leaning against reading table with scholarly elegance"
            ]
        },
        "rooftop_terrace": {
            "base": "luxurious rooftop terrace with city panorama and elegant outdoor furniture",
            "variations": ["penthouse terrace", "hotel rooftop", "private sky garden", "urban oasis", "rooftop lounge", "sky deck"],
            "activities": [
                "lounging on outdoor furniture with city lights backdrop",
                "posing by terrace railing with wind in hair",
                "sitting on outdoor sofa with legs elegantly positioned",
                "stretching under open sky with urban elegance",
                "dancing freely on terrace with city panorama",
                "adjusting outfit while enjoying rooftop breeze",
                "sitting on terrace edge with confident posture",
                "leaning against terrace furniture with relaxed confidence",
                "posing with urban skyline as dramatic backdrop",
                "walking gracefully across terrace with city views"
            ]
        },
        "designer_boutique": {
            "base": "high-end fashion boutique with luxury clothing displays and elegant interior",
            "variations": ["luxury fashion store", "designer showroom", "haute couture atelier", "vintage clothing boutique", "jewelry store", "accessories shop"],
            "activities": [
                "trying on elegant clothing with confidence",
                "posing by luxury clothing racks",
                "sitting on boutique furniture while shopping",
                "adjusting designer outfit in boutique mirror",
                "walking gracefully through boutique aisles",
                "leaning against boutique counter with shopping bags",
                "posing with luxury accessories confidently",
                "sitting in boutique changing area",
                "stretching while browsing clothing collections",
                "adjusting jewelry while standing by display case"
            ]
        }
    }

    # 50+ ЗОВНІШНОСТЕЙ З ЕТНІЧНІСТЮ
    appearances = [
        # Скандинавські
        "stunning 18-year-old Scandinavian blonde with innocent yet seductive charm and piercing blue eyes",
        "elegant 19-year-old Norwegian beauty with natural platinum hair and ethereal grace", 
        "captivating 20-year-old Swedish model with crystal blue eyes and porcelain skin",
        "alluring 21-year-old Danish woman with honey blonde hair and sophisticated charm",
        
        # Азійські
        "gorgeous 19-year-old Asian beauty with natural youthful elegance and flowing silky black hair",
        "sophisticated 22-year-old Japanese woman with delicate features and graceful movements",
        "stunning 24-year-old Korean beauty with flawless skin and elegant posture",
        "captivating 23-year-old Chinese model with mysterious charm and silky straight hair",
        "beautiful 25-year-old Thai woman with exotic appeal and natural grace",
        
        # Латинські
        "attractive 20-year-old Latina with playful energy, warm bronze skin and sensual curves",
        "seductive 23-year-old Mexican beauty with passionate eyes and natural rhythm",
        "stunning 25-year-old Brazilian with sensual appeal, golden tan and confident smile",
        "gorgeous 24-year-old Colombian model with vibrant energy and natural elegance",
        "captivating 22-year-old Argentine woman with sophisticated charm and graceful movements",
        
        # Східно-Європейські
        "beautiful 21-year-old Eastern European with confident allure and platinum blonde hair",
        "elegant 23-year-old Russian model with aristocratic features and mysterious charm",
        "stunning 25-year-old Ukrainian beauty with natural grace and confident presence",
        "gorgeous 24-year-old Polish woman with sophisticated elegance and warm smile",
        "captivating 22-year-old Czech model with artistic charm and natural poise",
        
        # Змішані раси
        "captivating 22-year-old mixed-race woman with exotic beauty and naturally graceful movements",
        "stunning 24-year-old Eurasian model with unique features and elegant confidence",
        "gorgeous 23-year-old Afro-Caribbean beauty with natural rhythm and warm charm",
        "beautiful 25-year-old Mediterranean woman with olive skin and sophisticated allure",
        
        # Європейські
        "seductive 23-year-old Italian brunette with mature confidence, olive skin and sultry gaze",
        "alluring 24-year-old French woman with sophisticated charm and chestnut hair cascade", 
        "gorgeous 26-year-old German blonde with professional elegance and crystal blue eyes",
        "elegant 25-year-old Spanish beauty with passionate charm and natural grace",
        "stunning 24-year-old Greek goddess with classical features and confident presence",
        
        # Інші
        "beautiful 23-year-old Australian surfer with natural tan and confident energy",
        "captivating 25-year-old Canadian model with fresh appeal and elegant posture",
        "stunning 24-year-old New Zealand beauty with natural charm and graceful movements"
    ]

    # РОЗШИРЕНІ ТИПИ ФІГУР
    body_types = [
        # ENORMOUS ГРУДИ (35%)
        "voluptuous bombshell with enormous natural H-cup breasts and perfect hourglass shape",
        "busty goddess with gigantic natural I-cup breasts practically spilling from any lingerie", 
        "curvy woman with massive natural J-cup breasts and ultra-feminine silhouette",
        "thick bombshell with huge natural G-cup breasts creating dramatic contrast with narrow waist",
        "busty beauty with enormous DD+ chest and naturally perfect feminine proportions",
        "curvaceous model with spectacular natural F-cup breasts and elegant posture",
        "voluptuous goddess with impressive natural E-cup breasts and confident bearing",
        
        # ENORMOUS СІДНИЦІ (35%)
        "thick booty model with enormous round ass and dramatically slim waist contrast",
        "PAWG goddess with gigantic bubble butt and naturally thick, powerful thighs",
        "curvy woman with massive ass cheeks and perfect hourglass proportions", 
        "bootylicious beauty with huge round buttocks and confidently seductive stride",
        "thick-bottomed goddess with enormous posterior and femininely curved hips",
        "curvaceous model with spectacular round ass and elegant leg positioning",
        "voluptuous woman with impressive posterior curves and natural confidence",
        
        # КОМБО ENORMOUS (20%)
        "ultra-curvy goddess with enormous natural H-cup breasts AND gigantic round ass",
        "thick bombshell with massive I-cup chest AND huge bubble butt proportions", 
        "voluptuous vixen combining gigantic natural breasts with enormous posterior curves",
        "curvaceous bombshell with spectacular bust AND impressive ass proportions",
        "ultra-feminine goddess with enormous chest AND massive posterior perfection",
        
        # Стандартні але привабливі (10%)
        "balanced hourglass figure with natural C-cup breasts and 24-inch waist",
        "athletic swimmer build with toned abs and naturally proportioned curves",
        "elegant model figure with natural B-cup breasts and graceful proportions",
        "fitness model build with toned curves and confident athletic posture",
        "dancer's body with natural flexibility and graceful feminine curves"
    ]

    # МЕГА-РОЗШИРЕНИЙ ОДЯГ
    clothing_states = [
        # Лінгері
        "wearing black French lace lingerie set with matching thigh-high stockings and garter belt",
        "in luxurious black silk bra and panties with intricate Venice lace details", 
        "wearing passionate red silk lingerie set with gold metallic chain accents",
        "in pure white bridal lingerie set with pearl and delicate lace accents",
        "wearing emerald green silk lingerie with golden chain details and cutouts",
        "in designer string bikini in metallic gold fabric with minimal coverage",
        "wearing violet silk lingerie with crystal embellishments and sheer panels",
        "in coral pink lace teddy with delicate floral patterns and ribbon ties",
        "wearing midnight blue satin lingerie set with silver metallic threading",
        
        # Елегантний одяг
        "wearing form-fitting little black dress that accentuates every curve perfectly",
        "in elegant white silk blouse with strategic buttons undone revealing cleavage",
        "wearing sophisticated pencil skirt with fitted blazer partially unbuttoned",
        "in flowing summer dress with plunging neckline and side slit details",
        "wearing designer cocktail dress with backless design and elegant draping",
        "in luxury evening gown with deep V-neck and body-hugging silhouette",
        "wearing chic business suit with fitted jacket emphasizing waist",
        "in romantic off-shoulder dress with delicate fabric and flowing design",
        
        # Casual але сексуальний
        "wearing oversized silk shirt partially unbuttoned revealing lace bra underneath", 
        "in fitted yoga outfit with sports bra and form-hugging leggings",
        "wearing denim shorts with cropped top showing toned midriff",
        "in silk pajama set with revealing camisole and tiny shorts",
        "wearing sheer black babydoll with flowing chiffon that reveals more than it hides",
        "in comfortable loungewear that somehow looks incredibly seductive",
        "wearing artistic wrap dress that accentuates curves while appearing casual"
    ]

    # НОВІ ДИНАМІЧНІ РУХИ ТА ПОЗИ
    dynamic_poses = [
        "gracefully adjusting hair while maintaining eye contact with camera",
        "slowly crossing and uncrossing legs while seated elegantly", 
        "gently stretching arms above head to emphasize natural curves",
        "confidently walking with natural hip sway and elegant posture",
        "playfully spinning with dress flowing around graceful silhouette",
        "sensually adjusting clothing straps with deliberate slow movements",
        "elegantly leaning forward while maintaining sophisticated composure",
        "gracefully turning to show profile while looking over shoulder",
        "confidently posing with hands on hips emphasizing waist",
        "naturally swaying to imaginary music with fluid body movements"
    ]

    # Генеруємо сценарій
    location_key = random.choice(list(locations.keys()))
    location_data = locations[location_key]
    location_variation = random.choice(location_data["variations"])
    
    appearance = random.choice(appearances)
    
    # 80% шансів на enormous
    enormous_chance = random.random()
    if enormous_chance < 0.8:
        enormous_body_types = [bt for bt in body_types if any(word in bt for word in ["enormous", "gigantic", "massive", "huge", "spectacular", "impressive"])]
        body_type = random.choice(enormous_body_types)
    else:
        standard_body_types = [bt for bt in body_types if not any(word in bt for word in ["enormous", "gigantic", "massive", "huge", "spectacular", "impressive"])]
        body_type = random.choice(standard_body_types)
    
    clothing = random.choice(clothing_states)
    activity = random.choice(location_data["activities"])
    
    # Додаємо динамічну позу
    dynamic_pose = random.choice(dynamic_poses)
    enhanced_activity = f"{activity}, {dynamic_pose}"
    
    return {
        "location": f"{location_variation} - {location_data['base']}",
        "activity": enhanced_activity,
        "appearance": appearance,
        "body_type": body_type,
        "clothing": clothing,
        "location_key": location_key,
        "dynamic_elements": dynamic_pose
    }


# --- New: Build an I2V video prompt that preserves the photo persona ---
def build_video_prompt_from_photo(persona: dict, part_a: str, part_b: str, motion: str = "slow cinematic dolly-in with gentle camera arc") -> str:
    """Compose a video prompt that reuses the exact persona/wardrobe from the T2I photo.

    - persona: dict with keys appearance/hair/wardrobe/pose/ethnicity
    - part_a/part_b: the two parts of the T2I prompt (photo), merged to preserve identity
    - motion: camera motion/behavior to add (no new persona or wardrobe)
    """
    appearance = str(persona.get("appearance", "")).strip()
    hair = str(persona.get("hair", "")).strip()
    wardrobe = str(persona.get("wardrobe", "")).strip()
    pose = str(persona.get("pose", "standing relaxed pose")).strip()

    identity_line = ", ".join([x for x in [appearance, hair, wardrobe, "full body, head-to-toe"] if x])
    behavior = ", ".join([
        "same person as the photo",
        pose,
        motion,
        "subtle natural movements, no jitter, smooth pacing",
    ])

    # Use both photo prompt parts to anchor the identity/style
    combined_photo_anchor = ", ".join([p for p in [part_a, part_b] if p])

    prompt = ", ".join([
        identity_line,
        combined_photo_anchor,
        behavior,
    ])
    return prompt
