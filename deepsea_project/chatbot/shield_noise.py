import torch
import librosa
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaLLM as Ollama
from langdetect import detect
from panns_inference import AudioTagging, labels
import panns_inference.config as panns_config

# ✅ Redirect to correct local CSV
panns_config.labels_csv_path = "class_labels_indices.csv"

# STEP 1: Load audio
def load_audio(file_path, sr=32000):
    waveform, _ = librosa.load(file_path, sr=sr, mono=True)
    return waveform[None, :]

# STEP 2: Classify marine sounds
def classify_audio(audio):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "Cnn14.pth")

    model = AudioTagging(checkpoint_path=model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    _, clipwise_output = model.inference(audio)

    top_indices = clipwise_output[0].argsort()[-5:][::-1]
    print(f"📊 Output shape: {clipwise_output.shape}")
    print(f"📄 Number of labels: {len(labels)}")

    valid_indices = [i for i in top_indices if i < len(labels)]
    return [(labels[i], clipwise_output[0, i].item()) for i in valid_indices]

# Translate model label into marine context
def translate_label_to_marine_context(label):
    marine_interpretations = {
        # Marine Contexts
        "Chant": "possibly rhythmic echolocation or sonar-like vocalization",
        "Rub": "frictional contact with equipment, nets, or seafloor",
        "Speech": "presence of divers or surface communication near hydrophone",
        "Music": "non-biological pattern, could be interference from surface vessel electronics",
        "Vehicle": "likely small boat or underwater remotely operated vehicle (ROV)",
        "Sine wave": "synthetic sonar tone or calibration signal",
        "Explosion": "detonation event, seismic airgun or underwater demolition",
        "Engine": "ship or vessel propulsion system noise",
        "Sonar": "active sonar pulse – potentially military or commercial use",
        "Ship": "large marine vessel acoustic signature (engine, propeller, hull noise)",
        "Propeller": "rotating propeller cavitation noise from boats or ships",
        "Bubble": "possible biological bubble feeding or turbulence",
        "Click": "biosonar click, likely from toothed whales (e.g., dolphins, sperm whales)",
        "Whistle": "long-duration cetacean call (e.g., dolphin whistle)",
        "Buzz": "foraging buzz from hunting cetaceans",
        "Humpback whale song": "mating vocalizations by humpback whales",
        "Echolocation": "high-frequency clicks for navigation, often odontocetes",
        "Ice": "moving ice, cracking, pressure ridges",
        "Rain": "surface precipitation noise",
        "Thunder": "low-frequency weather-related acoustic events",
        "Wind": "continuous surface agitation, noise floor elevation",
        "Seismic survey": "impulsive noise from airguns for geological mapping",
        "Fish": "chorus or individual fish grunts/pops – spawning activity or territory marking",
        "Snapping shrimp": "short, sharp clicks from snapping claws",
        "Hydrophone cable strum": "flow-induced vibration from instrument tether",
        "Chains": "anchor drag or gear deployment",
        "Dragging": "equipment being pulled on seafloor (e.g., trawl nets)",
        "Drilling": "rotary acoustic pattern from underwater drilling operations",
        "Construction": "pile driving, hammering, or mechanical work",
        "Beeping": "acoustic pinger or acoustic telemetry tag",
        "Alarm": "possibly artificial acoustic alarm used for navigation or safety",
        "Jet": "overhead aircraft or airport adjacent to hydrophone site",
        "Helicopter": "rotor blade modulation detected underwater (especially in shallow water)",
        "Train": "land-based transport noise coupling into water near shoreline",
        "Chainsaw": "vibratory industrial tools near water coupling acoustically",
        "Crowd": "anthropogenic presence on vessels or docks near hydrophone",
        "Radio": "non-acoustic contamination from device cross-talk (rare)",
        "Mechanical fan": "vessel cooling systems or deck-mounted HVAC",
        "Pump": "ballast pump or bilge operations on ships",
        "Motorboat": "small high-RPM craft, distinct from deep-draft vessel",
        "Turbine": "underwater energy systems (e.g., tidal turbines)",
        "Wave": "natural surface movement, rough seas",
        "Horn": "vessel signaling or port activity"
    }

    return marine_interpretations.get(label, f"Unclassified label: '{label}' (may require expert review)")

# STEP 3: Ask for context
def get_context():
    print("\n🌍 Please enter the environmental context:")
    location = input("📍 Location: ").strip()
    date = input("📅 Date (e.g., July 2024): ").strip()
    species_input = input("🐠 Species present (comma-separated): ").strip()
    species_list = [s.strip() for s in species_input.split(",") if s.strip()]

    return {
        "location": location or "Unknown location",
        "date": date or "Unknown date",
        "present_species": species_list or ["Unknown species"]
    }

# STEP 4: Build query
def build_user_query(tags, context, intensity="~85 dB"):
    tag_text = ', '.join([
        f"{tag} → {translate_label_to_marine_context(tag)} ({score:.2f})"
        for tag, score in tags
    ])
    species = ', '.join(context['present_species'])

    return f"""
Detected acoustic events:
{tag_text}

Estimated intensity: {intensity}
Location: {context['location']}
Date: {context['date']}
Nearby species: {species}

Note: All labels refer to underwater interpretations based on hydrophone data. Please analyze this as a marine acoustic expert and oceanographer.

What are the risks and recommended actions?
""".strip()

# STEP 5: LLM only
def generate_report(query):
    lang = detect(query)
    llm = Ollama(model="mistral", temperature=0.7, top_p=0.9, top_k=50)

    if lang == "en":

        prompt= f"""
        You are a senior oceanographer specializing in marine acoustics and underwater noise pollution, working within the Mediterranean region. Based on the hydrophone data and contextual information below, generate a concise **Marine Acoustic Risk Report** intended for environmental authorities, researchers, and coastal agencies in Tunisia.

        ---
        Hydrophone Acoustic and Environmental Context:
        {query}
        ---

        📄 Structure the report using these headings:
        1. **Summary of Acoustic Signatures**
        2. **Ecological and Oceanographic Risk Level** (Low / Moderate / High)
        3. **Impact on Marine Species and Coastal Ecosystems**
        4. **Recommendations for Monitoring and Mitigation in Tunisian Waters**

        🎯 Guidelines:
        - Use formal, scientific language appropriate for oceanographic research in the Mediterranean.
        - Be concise (no more than 100 words).
        - Consider vessel traffic, fishing activity, marine biodiversity, and coastal dynamics.
        - Ensure the report is suitable for institutions in Tunisia such as INSTM (Institut National des Sciences et Technologies de la Mer) or coastal management authorities.
        """
   
    else:
        prompt = f"""
        Vous êtes un océanographe senior spécialisé en acoustique sous-marine et en pollution sonore dans la région méditerranéenne. À partir des données hydroacoustiques et du contexte ci-dessous, rédigez un **rapport concis d’évaluation du risque acoustique marin** destiné aux autorités environnementales, chercheurs et agences côtières en Tunisie.

        ---
        Données acoustiques et contexte environnemental :
        {query}
        ---

        📄 Structure attendue :
        1. **Résumé des signatures acoustiques détectées**
        2. **Niveau de risque écologique et océanographique** (Faible / Modéré / Élevé)
        3. **Impact sur les espèces marines et les écosystèmes côtiers**
        4. **Recommandations pour la surveillance et l’atténuation dans les eaux tunisiennes**

        🎯 Consignes :
        - Utilisez un langage scientifique et professionnel adapté aux travaux océanographiques en Méditerranée.
        - Soyez concis (maximum 100 mots).
        - Tenez compte du trafic maritime, des activités de pêche, de la biodiversité marine et des dynamiques côtières.
        - Le rapport doit être pertinent pour les institutions tunisiennes telles que l’INSTM ou les agences de gestion du littoral.
        """

    return llm.invoke(prompt)
