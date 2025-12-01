"""
Synthesis-only implementation using langchain_core Runnable API.

- Uses detailed Z_POOL (scene_template now produces scenes only).
- scene_hint_mode supports "keep", "none", "auto".
- Uses langchain_core prompt | llm runnable composition instead of LLMChain.
- Stores full prompts and raw LLM outputs for traceability.

Note: If your installed langchain_core uses `.run()` instead of `.invoke()`, replace `.invoke(...)` calls accordingly.
"""
import csv
import os
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# ---- Use langchain_core (runnable API) ----
# Install: pip install langchain-core
#from langchain_core import PromptTemplate, OpenAI  # LLM wrapper and prompt template

from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI as OpenAI

# Runnable composition uses the '|' operator (prompt | llm) to form a runnable.
# The runnable is then invoked with .invoke() or .run() depending on your langchain_core version.

# Path to your uploaded proposal. The environment will transform this path to a URL as needed.
PROPOSAL_PDF = "/mnt/data/Modeling_novel_needs_generation_1.pdf"

# ------------------------
# Config
# ------------------------
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#if not OPENAI_API_KEY:
#    raise RuntimeError("Set OPENAI_API_KEY in your environment")

# Create the LLM object (OpenAI-compatible). If you use DeepSeek or other OpenAI-compatible provider,
# you can pass `api_key`/`base_url`/`model` here (see lib docs).
llm = OpenAI(api_key="",
            base_url="https://api.deepseek.com",
             model="deepseek-chat",
             temperature=0.9)  # you can pass api_key="...", base_url="...", model="..." if needed

# Product catalog — replace / expand for your use
P_products = [
    "folding electric bike",
    "portable solar charger",
    "digital photo frame",
    "smartwatch",
    #"telepresence robot rental",
    #"AI cooking assistant",
    #"edible cutlery subscription",
    #"micro-projector keychain",
    #"wearable air quality patch",
    #"noise-cancelling stroller cover",
]

products_1 = [
    "Smartphone",
    "Laptop computer",
    "Reusable water bottle",
    "Electric toothbrush",
    "Backpack",
    "Coffee maker",
    "Microwave oven",
    "Headphones / Earbuds",
    "Notebook / Planner",
    "Sneakers",
    "LED desk lamp",
    "Kitchen knife set",
    "Laundry detergent",
    "Smartwatch / Fitness tracker",
    "Vacuum cleaner",
    "Hair dryer",
    "Sunglasses",
    "Bluetooth speaker",
    "Portable power bank",
    "Thermal mug / Travel cup"
]

P_products.extend(products_1)
print("P_products length: ",len(P_products))

# User personas (U_pool)
U_pool = [
    {"label": "remote worker", "desc": "works from home; values productivity and comfort"},
    {"label": "commuter", "desc": "daily public transit user; time-pressed"},
    {"label": "eco-millennial", "desc": "environmentally conscious; tries new sustainable products"},
    {"label": "new parent", "desc": "has a newborn or infant; cares about safety and convenience"},
    {"label": "college student", "desc": "budget conscious, experimental, late-night routines"},
]

characters = [
    {"label": "Barista", "desc": "friendly, early-riser, creative drinks, multitasker, customer-focused"},
    {"label": "Office Manager", "desc": "organized, detail-oriented, logistics, calendar master, problem-solver"},
    {"label": "High-School Teacher", "desc": "patient, mentor, lesson-planner, adaptable, youth advocate"},
    {"label": "Freelance Designer", "desc": "creative, deadline-juggler, remote work, branding, self-motivated"},
    {"label": "City Bus Driver", "desc": "dependable, route expert, calm under pressure, community interaction"},
    {"label": "College Student", "desc": "curious, budget-conscious, social, overcaffeinated, career-searching"},
    {"label": "Local Shop Owner", "desc": "entrepreneurial, community-anchored, hands-on, customer service"},
    {"label": "Retired Grandparent", "desc": "wisdom, routine, hobbies, supportive, storyteller"},
    {"label": "Delivery Worker", "desc": "fast-paced, navigation expert, weather-proof, physically active"},
    {"label": "Healthcare Nurse", "desc": "compassionate, long shifts, medical knowledge, emotional resilience"},
    {"label": "Software Engineer", "desc": "analytical, problem-solver, tech stack, debugging, remote/office hybrid"},
    {"label": "Fitness Instructor", "desc": "energetic, motivational, health-focused, disciplined, upbeat"},
    {"label": "Single Parent", "desc": "resilient, multitasking, resourceful, caring, time-stretched"},
    {"label": "Restaurant Chef", "desc": "creative food, fast decision-making, precision, heat-tolerant, leadership"},
    {"label": "Taxi/Ride-share Driver", "desc": "conversational, navigation, flexible hours, people-watcher"},
    {"label": "Community Volunteer", "desc": "altruistic, event-oriented, outreach, teamwork, local impact"},
    {"label": "Journalist", "desc": "investigative, deadline-driven, communicator, news-aware, persistent"},
    {"label": "Environmental Activist", "desc": "passionate, sustainability, advocacy, grassroots movements"},
    {"label": "Librarian", "desc": "organized, quiet strength, information guide, research helper"},
    {"label": "Small-Town Mechanic", "desc": "hands-on, problem-solver, grease-smudged, trusted by locals"}
]

U_pool.extend(characters)
print("U_pool length: ",len(U_pool))

# ------------------------
# Corrected / detailed Z_POOL (scene_template generates scenes only)
# ------------------------
Z_POOL: List[Dict[str, Any]] = [
    {
        "label": "functional",
        "description": "Practical, utilitarian motivation: solves a concrete task, saves time/effort/cost, or improves performance.",
        "generation_prompt_hint": "Produce practical, efficiency-focused wording in the intention step.",
        "scene_template": (
            "Describe a realistic physical or temporal environment (place, time, conditions, constraints) "
            "where a {user_desc} might be present with the product '{product}'. "
            "Focus on setting details: location, time of day, environmental constraints, and contextual limitations. "
            "Keep the output around 150 words."
            #"Do NOT describe any product actions or user intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short, practical user intention in the form 'The user may intend/want to do ... with {product}' that describes a measurable or concrete goal."
        ),
        "keywords_signals": ["save time", "automate", "efficient"],
        "prior": 0.30
    },
    {
        "label": "emotional",
        "description": "Affective motivation: seeking mood change, comfort, enjoyment, nostalgia, stress relief or personal wellbeing.",
        "generation_prompt_hint": "Produce emotionally expressive wording in the intention step (comfort, nostalgia, mood).",
        "scene_template": (
            "Describe an emotionally colored environment involving the product '{product}' and a {user_desc}. "
            "Focus on sensory details (lighting, sounds, atmosphere), mood, and feelings that characterize the scene. "
            "Keep the output around 150 words."
            #"Do NOT describe any product usage or user intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short intention in the form 'The user may intend/want to do ... with {product}' that captures an emotional or experiential goal."
        ),
        "keywords_signals": ["comfort", "relax", "nostalgia"],
        "prior": 0.15
    },
    {
        "label": "social",
        "description": "Connection/status motivation: creating or maintaining social ties, sharing, belonging, social approval or group participation.",
        "generation_prompt_hint": "Emphasize sharing, connecting, or social signaling in the intention step.",
        "scene_template": (
            "Describe a scene where social presence matters — others are nearby or the context is group-focused — involving '{product}' and a {user_desc}. "
            "Focus on social layout, nearby actors, and social affordances (crowd, party, group setting). "
            "Keep the output around 150 words."
            #"Do NOT describe product usage or intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short intention in the form 'The user may intend/want to do ... with {product}' that involves sharing, connecting, or social outcomes."
        ),
        "keywords_signals": ["share", "friends", "group"],
        "prior": 0.15
    },
    {
        "label": "symbolic",
        "description": "Identity/expressive motivation: product use expresses identity, values, or personal style.",
        "generation_prompt_hint": "Emphasize identity expression or signaling in the intention wording.",
        "scene_template": (
            "Describe a scene rich in visual or cultural cues where identity or values are salient, involving '{product}' and a {user_desc}. "
            "Focus on visual style, symbolic artifacts, and cues that signal personal identity. "
            "Keep the output around 150 words."
            #"Do NOT describe usage or intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short intention in the form 'The user may intend/want to do ... with {product}' that expresses identity, style, or values."
        ),
        "keywords_signals": ["identity", "style", "signal"],
        "prior": 0.10
    },
    {
        "label": "safety",
        "description": "Protection and control: reduces risk, prevents harm, increases security or reliability.",
        "generation_prompt_hint": "Emphasize protection, monitoring, or prevention in the intention wording.",
        "scene_template": (
            "Describe a scene where safety, risk, or monitoring concerns are salient for a {user_desc} in presence of '{product}'. "
            "Focus on threats, vulnerabilities, or environmental factors that make safety relevant. "
            "Keep the output around 150 words."
            #"Do NOT describe product usage or intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short intention in the form 'The user may intend/want to do ... with {product}' that addresses safety, monitoring, or risk reduction."
        ),
        "keywords_signals": ["safety", "monitor", "protect"],
        "prior": 0.10
    },
    {
        "label": "exploratory",
        "description": "Curiosity and novelty-seeking: experimenting, learning, discovery, or trying new experiences.",
        "generation_prompt_hint": "Emphasize discovery, experimentation, or novelty in the intention.",
        "scene_template": (
            "Describe a scene that invites exploration or experimentation for a {user_desc} with '{product}'. "
            "Focus on openness, novelty, and conditions conducive to trying something new. "
            "Keep the output around 150 words."
            #"Do NOT describe product usage or intentions."
        ),
        "intention_template": (
            "Given the product '{product}', the scene described above, and the user persona '{user_desc}',\n"
            "write a short intention in the form 'The user may intend/want to do ... with {product}' that highlights experimentation or discovery."
        ),
        "keywords_signals": ["try", "experiment", "discover"],
        "prior": 0.10
    },
]

# ------------------------
# Scene hint list (for mode "keep")
# ------------------------
SCENE_HINTS = [
    "a rainy morning commute",
    "a late-night study session",
    "a cramped studio apartment",
    "a family weekend picnic",
    "a crowded city festival",
    "a short overnight trip",
    "a small office break room",
    "a quiet bedroom before sleep",
    "a busy grocery shopping run",
    "an outdoor hiking rest stop",
    "a noisy open-plan office",
    "a public transit evening ride",
    "a backyard barbecue",
    "a college dorm common area",
    "a long-haul red-eye flight"
]

# ------------------------
# Data structures
# ------------------------
@dataclass
class GenerationRecord:
    p: str
    u_label: str
    u_desc: str
    z_label: str
    z_desc: str
    scene_hint_mode: str
    scene_hint: str
    scene_prompt: str
    scene: str
    intention_prompt: str
    intention: str
    raw_llm_scene_output: str
    raw_llm_intent_output: str
    proposal_pdf_ref: str = PROPOSAL_PDF

# ------------------------
# Helpers
# ------------------------
def pick_user_persona(U_pool: List[Dict[str, str]]) -> Dict[str, str]:
    return random.choice(U_pool)

def pick_product(P_products: List[str]) -> str:
    return random.choice(P_products)

def pick_z_entry(Z_POOL: List[Dict[str, Any]]) -> Dict[str, Any]:
    priors = [z.get("prior", 1.0) for z in Z_POOL]
    total = sum(priors)
    probs = [p / total for p in priors]
    return random.choices(Z_POOL, weights=probs, k=1)[0]

# ------------------------
# Core generation (with scene_hint modes) using langchain_core runnables
# ------------------------
def generate_for_puz(
    p: str,
    u: Dict[str, str],
    z_entry: Dict[str, Any],
    llm,
    R_s: int = 3,
    R_t: int = 3,
    scene_hint_mode: str = "keep",  # "keep", "none", or "auto"
    temp_scene: float = 0.9,
    temp_intent: float = 0.9
) -> List[GenerationRecord]:
    """
    Generate scenes & intentions for product p, persona u, and detailed z_entry.
    scene_hint_mode:
      - "keep": sample scene_hint from SCENE_HINTS
      - "none": do not use a scene_hint (simpler prompt)
      - "auto": ask LLM to produce a brief scene_hint, then expand into a scene
    """
    records: List[GenerationRecord] = []

    # ---- Possibly produce or sample scene_hint ----
    if scene_hint_mode == "keep":
        scene_hint_source = random.choice(SCENE_HINTS)
    elif scene_hint_mode == "none":
        scene_hint_source = None
    elif scene_hint_mode == "auto":
        auto_hint_prompt = (
            f"Propose a short scene hint (one phrase) for imagining an environment where a {u['desc']} might be present "
            f"with the product '{p}'. Keep it short (1-6 words), concrete (e.g., 'late-night subway commute')."
        )
        auto_template = PromptTemplate(template=auto_hint_prompt, input_variables=[])
        auto_runnable = auto_template | llm
        try:
            auto_out = auto_runnable.invoke({})  # or .run() depending on version
            scene_hint_source = str(auto_out).strip().split("\n")[0]
        except Exception as e:
            scene_hint_source = None
    else:
        raise ValueError("scene_hint_mode must be one of 'keep', 'none', or 'auto'")

    # ---- Build scene prompt (fill z_entry.scene_template) ----
    # Replace placeholders in scene_template with product and user_desc
    #scene_template_text = z_entry["scene_template"].replace("{product}", p).replace("{user_desc}", u["desc"])
    user_label_desc=u["label"]+" who has characteristics of: "+u["desc"]
    scene_template_text = z_entry["scene_template"].replace("{product}", p).replace("{user_desc}", user_label_desc)
    if "{scene_hint}" in scene_template_text:
        scene_template_text = scene_template_text.replace("{scene_hint}", scene_hint_source or "")

    # Prepend generation_hint if present
    full_scene_prompt = (z_entry.get("generation_prompt_hint", "") + "\n\n" + scene_template_text).strip()

    # Create PromptTemplate and runnable (prompt | llm)
    scene_prompt_template = PromptTemplate(template=full_scene_prompt, input_variables=[])
    scene_runnable = scene_prompt_template | llm

    # Generate R_s scenes
    scene_outputs = []
    for _ in range(R_s):
        try:
            s_out = scene_runnable.invoke({})  # or .run()
        except Exception as e:
            s_out = f"[LLM scene generation error: {e}]"
        s_out_text = str(s_out.content).strip()
        scene_outputs.append(s_out_text)

    # ---- For each scene, generate intention(s) using intention_template runnable ----
    for scene_out in scene_outputs:
        #intent_template_text = z_entry["intention_template"].replace("{product}", p).replace("{user_desc}", u["desc"]).replace("{scene}", scene_out)
        user_label_desc=u["label"]+" who has characteristics of: "+u["desc"]
        intent_template_text = z_entry["intention_template"].replace("{product}", p).replace("{user_desc}", user_label_desc).replace("{scene}", scene_out)
        full_intent_prompt = (z_entry.get("generation_prompt_hint", "") + "\n\n" + f"Scene (context): {scene_out}\n\n" + intent_template_text).strip()

        intent_prompt_template = PromptTemplate(template=full_intent_prompt, input_variables=[])
        intent_runnable = intent_prompt_template | llm

        for _ in range(R_t):
            try:
                t_out = intent_runnable.invoke({})  # or .run()
            except Exception as e:
                t_out = f"[LLM intent generation error: {e}]"
            intent_text = str(t_out.content).strip()

            rec = GenerationRecord(
                p=p,
                u_label=u["label"],
                u_desc=u["desc"],
                z_label=z_entry["label"],
                z_desc=z_entry["description"],
                scene_hint_mode=scene_hint_mode,
                scene_hint=scene_hint_source or "",
                scene_prompt=full_scene_prompt,
                scene=scene_out,
                intention_prompt=full_intent_prompt,
                intention=intent_text,
                raw_llm_scene_output=scene_out,
                raw_llm_intent_output=t_out
            )
            records.append(rec)

    return records

# ------------------------
# Batch synthesis driver
# ------------------------
"""
def synthesize_dataset(
    num_rounds: int = 100,
    R_s: int = 3,
    R_t: int = 3,
    scene_hint_mode: str = "keep"
) -> List[Dict[str, Any]]:
    all_records: List[GenerationRecord] = []
    for _ in range(num_rounds):
        p = pick_product(P_products)
        u = pick_user_persona(U_pool)
        z_entry = pick_z_entry(Z_POOL)
        recs = generate_for_puz(
            p, u, z_entry, llm,
            R_s=R_s, R_t=R_t,
            scene_hint_mode=scene_hint_mode
        )
        all_records.extend(recs)
    return [asdict(r) for r in all_records]
"""
def synthesize_dataset(
    num_rounds: int = 100,
    R_s: int = 3,
    R_t: int = 3,
    scene_hint_mode: str = "keep",
    output_csv_path: str = "result/5tuple_needs/needs_generated_sample.csv"
) -> List[Dict[str, Any]]:

    # If file does not exist, create header row
    file_exists = os.path.isfile(output_csv_path)
    if not file_exists:
        with open(output_csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "p", "u_label", "u_desc", "z_label", "z_desc",
                "scene_hint_mode", "scene_hint",
                "scene_prompt", "scene",
                "intention_prompt", "intention",
                "raw_llm_scene_output", "raw_llm_intent_output",
                "proposal_pdf_ref"
            ])

    all_records: List[GenerationRecord] = []

    for round_idx in range(num_rounds):

        p = pick_product(P_products)
        u = pick_user_persona(U_pool)
        z_entry = pick_z_entry(Z_POOL)

        recs = generate_for_puz(
            p, u, z_entry, llm,
            R_s=R_s, R_t=R_t,
            scene_hint_mode=scene_hint_mode
        )

        # store in memory list
        all_records.extend(recs)

        # append to CSV immediately
        with open(output_csv_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in recs:
                rd = asdict(r)
                writer.writerow([
                    rd["p"],
                    rd["u_label"], rd["u_desc"],
                    rd["z_label"], rd["z_desc"],
                    rd["scene_hint_mode"], rd["scene_hint"],
                    rd["scene_prompt"], rd["scene"],
                    rd["intention_prompt"], rd["intention"],
                    rd["raw_llm_scene_output"], rd["raw_llm_intent_output"],
                    rd["proposal_pdf_ref"]
                ])

        print(f"[Round {round_idx+1}/{num_rounds}] Wrote {len(recs)} samples to {output_csv_path}")

    return [asdict(r) for r in all_records]


# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    SAMPLE_ROUNDS = 20
    R_S = 1
    R_T = 1
    MODE = "none"   # choose "keep", "none", or "auto"

    print("Starting generation (LLM calls via langchain_core runnables)...")
    #assert(1==2)
    generated = synthesize_dataset(num_rounds=SAMPLE_ROUNDS, R_s=R_S, R_t=R_T, scene_hint_mode=MODE)
    print(f"Produced {len(generated)} generated records. Showing 3 examples:\n")
    for rec in generated[:3]:
        print("=== RECORD ===")
        print("Product:", rec["p"])
        print("User:", rec["u_label"], "-", rec["u_desc"])
        print("Z (need-type):", rec["z_label"])
        print("scene_hint_mode:", rec["scene_hint_mode"])
        print("scene_hint:", rec["scene_hint"])
        print("Scene prompt (truncated):", rec["scene_prompt"][:320], "...")
        print("Scene (LLM):", rec["scene"])
        print("Intention prompt (truncated):", rec["intention_prompt"][:320], "...")
        print("Intention (LLM):", rec["intention"])
        print("Proposal PDF ref:", rec["proposal_pdf_ref"])
        print()
