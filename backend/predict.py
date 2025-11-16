import os
from ocr_extract import extract_text  # type: ignore
import torch  # type: ignore
from torchvision import transforms  # type: ignore
from PIL import Image  # type: ignore

HARMFUL_FILE = os.path.join(os.path.dirname(__file__), "harmful_ingredients.txt")
if os.path.exists(HARMFUL_FILE):
    with open(HARMFUL_FILE, encoding="utf-8") as f:
        HARMFUL = [line.strip() for line in f if line.strip()]
else:
    HARMFUL = [
    # Oils & Fats
    "Palm Oil", "Repeatedly Heated Palm Oil", "Hydrogenated Oil",
    "Partially Hydrogenated Oils", "Shortening", "Vanaspati",
    "Interesterified Fats", "Trans Fats",
    
    # Sugars
    "Sugar", "Refined Sugar", "Liquid Glucose",
    "High Fructose Corn Syrup", "Corn Syrup Solids",
    
    # Sweeteners
    "Aspartame", "Sucralose", "Acesulfame K", "Saccharin",
    "Neotame", "Advantame",
    
    # Artificial Colors
    "Tartrazine (E102)", "Sunset Yellow (E110)", "Carmoisine (E122)",
    "Ponceau 4R (E124)", "Allura Red (E129)", "Brilliant Blue (E133)",
    "Fast Green (E143)", "Caramel Color (E150d)",
    
    # Preservatives
    "Sodium Benzoate", "Potassium Sorbate", "Calcium Sorbate",
    "Benzoic Acid", "Sulphites (E220–E228)", "Propyl Gallate",
    "BHA", "BHT", "Sodium Nitrate", "Sodium Nitrite",
    "Potassium Bromate", "Calcium Propionate",
    "TBHQ", "EDTA", "Formaldehyde contamination",
    
    # Flavor Enhancers
    "Monosodium Glutamate (MSG)", "Disodium Inosinate (E631)",
    "Disodium Guanylate (E627)", "Yeast Extract (Excess)",
    "Artificial Flavors", "Smoke Flavor Additives",
    
    # Emulsifiers
    "Mono & Diglycerides (E471)", "Polysorbate 80 (E433)",
    "Sorbitan Monostearate (E491)", "Synthetic Lecithin",
    "PGPR (E476)", "DATEM (E472e)",
    
    # Stabilizers / Thickeners
    "Carrageenan (E407)", "Xanthan Gum (E415)", "Guar Gum (E412)",
    "Locust Bean Gum (E410)", "CMC (E466)", "Gellan Gum (E418)",
    
    # Acidity Regulators
    "Phosphoric Acid", "Malic Acid (Synthetic)", "Lactic Acid (Synthetic)",
    "Citric Acid (Overuse)",
    
    # Anti-Caking / Bulking
    "Silicon Dioxide", "Calcium Silicate", "Talc (E553b)",
    "Magnesium Stearate",
    
    # Flour / Bakery Additives
    "Refined Flour (Maida)", "Azodicarbonamide (ADA)",
    "Potassium Bromate", "Benzoyl Peroxide",
    
    # Beverage Additives
    "Excess Caffeine", "Excess Carbonation (CO2)",
    
    # Contaminants
    "Acrylamide (from frying)", "PAHs (burned oils)", "3-MCPD",
    "Glycidyl Esters",
    
    # Packaging Chemicals
    "BPA", "BPS", "Phthalates", "Microplastics",
    
    # General Excess Nutrients
    "Excess Sodium", "Excess Saturated Fat"
]


HIGH_RISK_INGREDIENTS = {
    "Trans Fats": "Avoid completely – major cause of heart disease",
    "Hydrogenated Oil": "Avoid completely – contains trans fats",
    "Partially Hydrogenated Oils": "Avoid completely – contains trans fats",
    "BHA": "Avoid completely – potential carcinogen",
    "BHT": "Avoid completely – potential carcinogen",
    "TBHQ": "Avoid – strong evidence of toxicity at high doses",
    "Potassium Bromate": "Avoid – banned in many countries, cancer risk",
    "Azodicarbonamide": "Avoid – asthma-causing dough improver",
    "Sodium Nitrite": "Avoid – can form carcinogenic nitrosamines",
    "Sodium Nitrate": "Avoid – cancer risk",
    "Acrylamide": "Avoid – formed in fried foods, linked to cancer",
    "3-MCPD": "Avoid – contaminant formed during oil refining",
    "Glycidyl Esters": "Avoid – known carcinogenic contaminant",
    "Artificial Colors (E102, E110, E122, E124, E129)": "Avoid for children – hyperactivity, allergy risk"
}


MEDIUM_RISK_INGREDIENTS = {
    "Palm Oil": "Moderation – inflammatory when overheated",
    "Sugar": "Moderation – obesity, diabetes risk",
    "Refined Sugar": "Moderation – empty calories",
    "High Fructose Corn Syrup": "Moderation – linked to metabolic disorder",
    "Liquid Glucose": "Moderation – high glycemic index",
    "Sodium Benzoate": "Moderation – forms benzene with Vitamin C",
    "Potassium Sorbate": "Moderation – safe but can irritate skin",
    "Monosodium Glutamate (MSG)": "Moderation – safe for most, sensitive for some",
    "Artificial Flavors": "Moderation – unknown long-term effects",
    "Caramel Color (E150d)": "Moderation – contains 4-MEI (possible carcinogen)",
    "Polysorbate 80": "Moderation – gut microbiome disruption",
    "Carrageenan": "Moderation – may cause gut inflammation in some people"
}


LOW_RISK_INGREDIENTS = {
    "Xanthan Gum": "Generally safe – used as stabilizer",
    "Lecithin": "Generally safe – natural emulsifier",
    "Citric Acid": "Safe unless consumed in high amounts",
    "Guar Gum": "Safe – natural fiber but causes bloating in excess",
    "Gellan Gum": "Safe – widely used stabilizer"
}

_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "food_safety_model.pth")
_image_model = None
def load_image_model():
    global _image_model
    if _image_model is None and os.path.exists(MODEL_PATH):
        try:
            _image_model = torch.load(MODEL_PATH, map_location="cpu")
            _image_model.eval()
        except:
            _image_model = None
    return _image_model

def detect_harmful_by_text(ocr_text: str):
    """Detect harmful ingredients and categorize by risk level."""
    found_high = []
    found_medium = []
    found_low = []
    
    low = ocr_text.lower()
    
    # Check high risk
    for ingredient, warning in HIGH_RISK_INGREDIENTS.items():
        if ingredient.lower() in low:
            found_high.append({"name": ingredient, "risk": "HIGH", "advice": warning})
    
    # Check medium risk
    for ingredient, warning in MEDIUM_RISK_INGREDIENTS.items():
        if ingredient.lower() in low:
            found_medium.append({"name": ingredient, "risk": "MEDIUM", "advice": warning})
    
    # Check low risk
    for ingredient, warning in LOW_RISK_INGREDIENTS.items():
        if ingredient.lower() in low:
            found_low.append({"name": ingredient, "risk": "LOW", "advice": warning})
    
    return found_high, found_medium, found_low

def get_health_recommendation(high_risk, medium_risk, low_risk):
    """Generate overall health recommendation based on detected ingredients."""
    if high_risk:
        return {
            "recommendation": "🔴 AVOID THIS PRODUCT",
            "details": f"Contains {len(high_risk)} high-risk ingredient(s) that should be avoided.",
            "severity": "critical"
        }
    elif medium_risk and len(medium_risk) >= 3:
        return {
            "recommendation": "🟠 NOT RECOMMENDED - High Risk",
            "details": f"Contains {len(medium_risk)} ingredients that are harmful if consumed regularly.",
            "severity": "high"
        }
    elif medium_risk:
        return {
            "recommendation": "🟡 CONSUME SPARINGLY",
            "details": f"Contains {len(medium_risk)} ingredient(s) that should be consumed in moderation.",
            "severity": "medium"
        }
    else:
        return {
            "recommendation": "🟢 RELATIVELY SAFE",
            "details": "No major harmful ingredients detected. Can be consumed occasionally.",
            "severity": "low"
        }

def predict(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    ocr_text = extract_text(image_path)
    high_risk, medium_risk, low_risk = detect_harmful_by_text(ocr_text)
    
    # Determine status
    if high_risk:
        status = "UNSAFE"
    elif medium_risk and len(medium_risk) >= 2:
        status = "UNSAFE"
    else:
        status = "SAFE"

    img_model = load_image_model()
    image_pred_is_unsafe = False
    if img_model:
        try:
            img = Image.open(image_path).convert("RGB")
            x = _transform(img).unsqueeze(0)
            with torch.no_grad():
                out = img_model(x)
                if isinstance(out, torch.Tensor) and out.dim() == 2:
                    _, pred = torch.max(out, 1)
                    image_pred_is_unsafe = int(pred.item()) == 1
        except:
            image_pred_is_unsafe = False

    if image_pred_is_unsafe:
        status = "UNSAFE"
    
    # Get recommendation
    recommendation = get_health_recommendation(high_risk, medium_risk, low_risk)
    
    return {
        "status": status,
        "harmful_high": high_risk,
        "harmful_medium": medium_risk,
        "harmful_low": low_risk,
        "recommendation": recommendation,
        "ocr_text": ocr_text
    }
