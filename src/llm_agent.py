import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

CHEST_PAIN_MAP = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
ECG_MAP = {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}
SLOPE_MAP = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
THAL_MAP = {0: "Normal", 1: "Fixed Defect", 2: "Reversable Defect", 3: "Unknown"}


def build_patient_context(patient: dict, prediction: dict, flags: list) -> str:
    sex = "Male" if patient.get("sex") == 1 else "Female"
    cp = CHEST_PAIN_MAP.get(patient.get("cp", 0), "Unknown")
    ecg = ECG_MAP.get(patient.get("restecg", 0), "Unknown")
    slope = SLOPE_MAP.get(patient.get("slope", 0), "Unknown")
    thal = THAL_MAP.get(patient.get("thal", 0), "Unknown")
    exang = "Yes" if patient.get("exang") == 1 else "No"
    fbs = "Yes (>120 mg/dl)" if patient.get("fbs") == 1 else "No"

    risk_pct = round(prediction.get("probability", 0) * 100, 1)
    risk_label = prediction.get("risk_label", "Unknown")

    flag_lines = "\n".join(f"- [{lvl.upper()}] {msg}" for lvl, msg in flags)

    return f"""PATIENT CLINICAL DATA:
- Age: {patient.get('age')} years, Sex: {sex}
- Resting Blood Pressure: {patient.get('trestbps')} mmHg
- Serum Cholesterol: {patient.get('chol')} mg/dl
- Fasting Blood Sugar >120 mg/dl: {fbs}
- Resting ECG: {ecg}
- Maximum Heart Rate Achieved: {patient.get('thalach')} bpm
- Exercise-Induced Angina: {exang}
- ST Depression (oldpeak): {patient.get('oldpeak')}
- Slope of ST Segment: {slope}
- Major Vessels Coloured: {patient.get('ca')}
- Thalassemia: {thal}
- Chest Pain Type: {cp}

ML MODEL PREDICTION:
- Risk Level: {risk_label}
- Risk Probability: {risk_pct}%

CLINICAL FLAGS (rule-based checks):
{flag_lines}"""


def build_prompt(patient_context: str, user_query: str) -> str:
    system = (
        "You are a health education assistant. You help patients understand their "
        "cardiovascular risk assessment results in simple, clear language. "
        "You do NOT diagnose or prescribe. Always recommend consulting a doctor. "
        "Base your response strictly on the patient data provided — do not invent values. "
        "Structure your response with these exact section headers:\n\n"
        "## Patient Risk Summary\n"
        "## Key Contributing Factors\n"
        "## Preventive Recommendations\n"
        "## When to Seek Medical Attention\n"
        "## References\n"
        "## Medical Disclaimer\n\n"
        "Keep each section concise and easy to understand. Use bullet points where helpful."
    )

    user = (
        f"{patient_context}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        "Please provide a structured health education report based on the data above."
    )

    return system, user


def call_groq(patient_context: str, user_query: str) -> dict:
    if not GROQ_API_KEY:
        return {"success": False, "text": "", "error": "No API key found"}

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        system_prompt, user_prompt = build_prompt(patient_context, user_query)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        text = response.choices[0].message.content
        return {"success": True, "text": text, "error": None}

    except Exception as e:
        return {"success": False, "text": "", "error": str(e)}


def parse_report_sections(text: str) -> dict:
    sections = {
        "risk_summary":       "",
        "contributing":       "",
        "recommendations":    "",
        "seek_attention":     "",
        "references":         "",
        "disclaimer":         "",
    }

    headers = {
        "## Patient Risk Summary":           "risk_summary",
        "## Key Contributing Factors":       "contributing",
        "## Preventive Recommendations":     "recommendations",
        "## When to Seek Medical Attention":  "seek_attention",
        "## References":                     "references",
        "## Medical Disclaimer":             "disclaimer",
    }

    current_key = None
    buffer = []

    for line in text.split("\n"):
        matched = False
        for header, key in headers.items():
            if line.strip().startswith(header.strip()):
                if current_key:
                    sections[current_key] = "\n".join(buffer).strip()
                current_key = key
                buffer = []
                matched = True
                break
        if not matched and current_key:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def generate_fallback_report(patient: dict, prediction: dict, flags: list) -> dict:
    risk_pct = round(prediction.get("probability", 0) * 100, 1)
    risk_label = prediction.get("risk_label", "Unknown")

    danger_flags = [msg for lvl, msg in flags if lvl == "danger"]
    warning_flags = [msg for lvl, msg in flags if lvl == "warning"]

    contributing_lines = []
    for msg in danger_flags:
        contributing_lines.append(f"- 🔴 {msg}")
    for msg in warning_flags:
        contributing_lines.append(f"- 🟡 {msg}")
    if not contributing_lines:
        contributing_lines = ["- No major risk factors flagged by clinical rules."]

    return {
        "risk_summary": (
            f"The ML model estimates a **{risk_pct}%** probability of cardiovascular disease, "
            f"categorised as **{risk_label}**. This is based on 13 clinical parameters analysed "
            "by a machine learning classifier trained on the UCI Heart Disease dataset."
        ),
        "contributing": "\n".join(contributing_lines),
        "recommendations": (
            "- Maintain a heart-healthy diet low in saturated fats and sodium.\n"
            "- Aim for at least 150 minutes of moderate exercise per week.\n"
            "- Monitor blood pressure and cholesterol regularly.\n"
            "- Avoid smoking and limit alcohol intake.\n"
            "- Manage stress through adequate sleep and relaxation."
        ),
        "seek_attention": (
            "- If you experience chest pain, shortness of breath, or palpitations — seek emergency care immediately.\n"
            "- Schedule a cardiology review if your risk score is Medium or High.\n"
            "- Follow up with your GP for routine monitoring of your vitals."
        ),
        "references": (
            "- American Heart Association: https://www.heart.org\n"
            "- UCI Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease\n"
            "- WHO Cardiovascular Risk Guidelines: https://www.who.int/health-topics/cardiovascular-diseases"
        ),
        "disclaimer": (
            "This report is generated for **educational purposes only** and does not constitute "
            "medical advice or diagnosis. It must not be used to make clinical decisions. "
            "Always consult a qualified healthcare professional for medical guidance."
        ),
    }


def run_agent(patient: dict, prediction: dict, flags: list, user_query: str) -> dict:
    state = {
        "step": "idle",
        "patient_context": "",
        "llm_text": "",
        "sections": {},
        "used_llm": False,
        "error": None,
    }

    state["step"] = "building_context"
    state["patient_context"] = build_patient_context(patient, prediction, flags)

    state["step"] = "calling_llm"
    if GROQ_API_KEY:
        result = call_groq(state["patient_context"], user_query)
        if result["success"]:
            state["llm_text"] = result["text"]
            state["sections"] = parse_report_sections(result["text"])
            state["used_llm"] = True
        else:
            state["error"] = result["error"]
            state["sections"] = generate_fallback_report(patient, prediction, flags)
    else:
        state["sections"] = generate_fallback_report(patient, prediction, flags)

    state["step"] = "done"
    return state
