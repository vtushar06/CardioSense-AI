import json
from typing import Dict, Any, List
from langchain_core.tools import tool
from .predictor import predict_single
from .explainer import get_rule_based_flags
from .vector_store import VectorStoreManager

@tool
def get_risk_prediction(patient_json: str) -> str:
    """
    Calculates the cardiovascular risk score using the trained ML model.
    Input should be a JSON string of patient features (age, sex, chol, etc.).
    """
    try:
        patient_data = json.loads(patient_json)
        result = predict_single(patient_data)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error calculating risk: {str(e)}"

@tool
def get_clinical_flags(patient_json: str) -> str:
    """
    Analyzes patient vitals against deterministic clinical thresholds (e.g., BP > 140).
    Input should be a JSON string of patient features.
    """
    try:
        patient_data = json.loads(patient_json)
        flags = get_rule_based_flags(patient_data)
        # Format as a clean string
        formatted_flags = [f"[{level.upper()}] {msg}" for level, msg in flags]
        return "\n".join(formatted_flags)
    except Exception as e:
        return f"Error analyzing flags: {str(e)}"

@tool
def search_medical_guidelines(query: str) -> str:
    """
    Searches the cardiovascular knowledge base for clinical guidelines and recommendations.
    Use this to find standard thresholds (BP, Cholesterol) or preventive advice.
    """
    try:
        manager = VectorStoreManager()
        results = manager.search(query, k=3)
        if not results:
            return "No relevant guidelines found in the knowledge base."
        
        context = []
        for doc in results:
            header = doc.metadata.get('Header 2', doc.metadata.get('Header 1', 'Guideline'))
            context.append(f"### {header}\n{doc.page_content}")
            
        return "\n\n".join(context)
    except Exception as e:
        return f"Error searching guidelines: {str(e)}"

# Collection of tools for the agent
CARDIO_TOOLS = [get_risk_prediction, get_clinical_flags, search_medical_guidelines]
