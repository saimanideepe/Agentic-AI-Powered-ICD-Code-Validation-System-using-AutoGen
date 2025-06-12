import os
import re
import json
from typing import List, Dict, Tuple, Any
from autogen import AssistantAgent, UserProxyAgent
from pydantic import BaseModel, ValidationError
import simple_icd_10 as icd
from groq import Groq
import numpy as np

from prompt import VALIDATION_PROMPT_TEMPLATE, ALTERNATIVE_SUGGESTION_PROMPT_TEMPLATE, CONFIDENCE_PROMPT_TEMPLATE

OPENAI_API_KEY = ""
GROQ_API_KEY = ""

groq_client = Groq(api_key=GROQ_API_KEY)

class GroqAgent:
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    def generate_reply(self, messages: List[Dict[str, str]]) -> str:
        response = groq_client.chat.completions.create(
            messages=messages,
            model=self.model,
        )
        return response.choices[0].message.content

openai_agent = AssistantAgent(
    name="OpenAI_Agent",
    llm_config={"api_key": OPENAI_API_KEY, "model": "gpt-4o"},
)
mistral_agent = GroqAgent(name="Groq_Mistral_Agent", model="mixtral-8x7b-32768")
llama_agent   = GroqAgent(name="Groq_LLaMA_Agent", model="llama3-70b-8192")
user_proxy    = UserProxyAgent(name="User", code_execution_config={"use_docker": False})

class RAGOutput(BaseModel):
    finalCodes: List[str]
    content: List[Dict[str, Any]]

def extract_icd_codes(response: str) -> List[str]:
    try:
        structured_response = RAGOutput.parse_raw(response)
        return structured_response.finalCodes
    except ValidationError:
        return re.findall(r'\b[A-Z]\d{1,2}\.?\d*\b', response)

def get_icd_descriptions(icd_codes: List[str]) -> Dict[str, str]:
    descriptions = {}
    for code in icd_codes:
        try:
            descriptions[code] = icd.get_description(code)
        except ValueError:
            descriptions[code] = "Description not found"
    return descriptions

def clean_json_response(response: str) -> str:
    response = response.strip()
    if response.startswith("```"):
        parts = response.split("\n")
        if parts[0].strip().startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip() == "```":
            parts = parts[:-1]
        response = "\n".join(parts).strip()
    return response

def validate_icd_codes(agent, icd_codes: List[str], descriptions: Dict[str, str], summary: str, max_retries: int = 2) -> Tuple[List[str], bool]:
    validated_codes = icd_codes.copy()
    retries = 0
    print(f"\n🔍 ==== Validation Started for {agent.name} ====")
    while retries < max_retries:
        confirmed_codes = []
        rejected_codes = []
        for code in validated_codes:
            desc = descriptions.get(code, "Description not found")
            prompt = VALIDATION_PROMPT_TEMPLATE.replace("{ICD_CODE}", code)\
                                                 .replace("{DESCRIPTION}", desc)\
                                                 .replace("{SUMMARY}", summary)
            print(f"\n🔵 {agent.name} validating ICD code: {code}")
            response = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
            if "CONFIRMED" in response.upper():
                print(f"✅ {agent.name} confirmed ICD code: {code}")
                confirmed_codes.append(code)
            else:
                rejected_codes.append(code)
                print(f"❌ {agent.name} rejected ICD code: {code} and suggested an alternative.")
        if not rejected_codes:
            print(f"\n✅ All ICD codes confirmed by {agent.name}")
            return confirmed_codes, True
        alt_prompt = ALTERNATIVE_SUGGESTION_PROMPT_TEMPLATE.replace("{PREVIOUS_CODES}", ', '.join(validated_codes))\
                                                             .replace("{SUMMARY}", summary)
        print(f"\n🔄 {agent.name} is suggesting alternative ICD codes...")
        response = agent.generate_reply(messages=[{"role": "user", "content": alt_prompt}])
        new_codes = extract_icd_codes(response)
        if not new_codes:
            print(f"⚠️ {agent.name} did not provide new codes. Retaining last confirmed codes.")
            return confirmed_codes, True
        print(f"🔄 {agent.name} suggested new ICD codes: {new_codes}")
        validated_codes = new_codes
        descriptions = get_icd_descriptions(validated_codes)
        retries += 1
    print(f"\n⚠️ Maximum retries reached for {agent.name}. Returning best available codes.")
    return validated_codes, False

def get_confidence_and_evidence(agent, icd_code: str, description: str, summary: str, max_retries: int = 2, default_score: int = 50, default_evidence: List[str] = None) -> Dict[str, Any]:
    if default_evidence is None:
        default_evidence = ["No evidence provided"]
    attempt = 0
    while attempt <= max_retries:
        if attempt == 0:
            prompt = CONFIDENCE_PROMPT_TEMPLATE.replace("{ICD_CODE}", icd_code)\
                                                 .replace("{DESCRIPTION}", description)\
                                                 .replace("{SUMMARY}", summary)
        else:
            prompt = "Dear Expert, your previous response did not follow the required JSON format. Please re-evaluate and provide a JSON with keys 'score' and 'evidence'."
        response = agent.generate_reply(messages=[{"role": "user", "content": prompt}])
        print(f"[DEBUG] Raw response from {agent.name}: {response}")
        try:
            cleaned_response = clean_json_response(response)
            data = json.loads(cleaned_response)
            score = int(data.get("score", default_score))
            evidence = data.get("evidence", default_evidence)
            if 0 <= score <= 100 and isinstance(evidence, list) and evidence:
                return {"score": score, "evidence": evidence}
        except Exception as e:
            print(f"[DEBUG] JSON parsing error from {agent.name}: {e}")
        attempt += 1
    return {"score": default_score, "evidence": default_evidence}

# --- Per-Model Processing Function ---
def process_model_icd_codes(rag_output: Dict[str, Any], agent, num_codes: int = 5) -> List[Dict[str, Any]]:
    initial_codes = rag_output.get("finalCodes", [])[:num_codes]
    joint_summary = "\n".join([entry.get("summary", "") for entry in rag_output.get("content", [])])
    descriptions = get_icd_descriptions(initial_codes)
    validated_codes, _ = validate_icd_codes(agent, initial_codes, descriptions, joint_summary, max_retries=3)
    final_codes = validated_codes[:num_codes]
    final_results = []
    for code in final_codes:
        desc = get_icd_descriptions([code])[code]
        confidence_result = get_confidence_and_evidence(agent, code, desc, joint_summary)
        final_results.append({
            "icd_code": code,
            "confidence": confidence_result["score"],
            "evidence": confidence_result["evidence"]
        })
    return final_results

def process_all_models_icd_codes(openai_rag_output: Dict[str, Any],
                                 mistral_rag_output: Dict[str, Any],
                                 llama_rag_output: Dict[str, Any]) -> Dict[str, Any]:
    print("\n========== Processing OpenAI RAG Output ==========")
    openai_results = process_model_icd_codes(openai_rag_output, openai_agent)
    
    print("\n========== Processing Mistral RAG Output ==========")
    mistral_results = process_model_icd_codes(mistral_rag_output, mistral_agent)
    
    print("\n========== Processing LLaMA RAG Output ==========")
    llama_results = process_model_icd_codes(llama_rag_output, llama_agent)

    final_results = {
        "OpenAI": openai_results,
        "Mistral": mistral_results,
        "LLaMA": llama_results
    }
    return final_results

example_openai_rag_output = {
  "finalCodes": ["E11.9", "I63.9"],
  "content": [
    {
      "disease": "Atrial fibrillation",
      "summary": "Dear Ms. Savannah Green,\n\nThank you for providing the updated medical report. Based on the new information, there is no mention of Atrial fibrillation or any related symptoms, treatments, or medications. Therefore, I cannot provide any specific information about this condition based on the provided context. However, I can provide some general information about Atrial fibrillation.\n\nAtrial fibrillation is a type of cardiac arrhythmia that is caused by a disorganized electrical activity in the atria, leading to an abnormal heart rhythm. Symptoms of Atrial fibrillation can include palpitations, shortness of breath, and fatigue. Treatment for Atrial fibrillation may include medications such as beta blockers, antiarrhythmic drugs, or cardioversion, as well as catheter ablation or surgical maze procedure to restore a normal heart rhythm.\n\nAs for the most relevant ICD-10 codes for Atrial fibrillation based on the provided medical report, there are none. The report does not mention any symptoms, treatments, or medications related to Atrial fibrillation. Here is an empty list of relevant ICD-10 codes for Atrial fibrillation based on the provided medical report:\n\n* None\n\nPlease note that there are no mentions of Atrial fibrillation or any related symptoms, treatments, or medications in the provided medical report. If you have any further questions or concerns, please do not hesitate to ask.\n\nBest regards,\n[Your Name]"
    },
    {
      "disease": "Hypertension",
      "summary": "Based on the provided context, here is a list of possible diseases that may be present:\n\n1. Hypertension (ICD-10 code: I10)\n2. Stroke (ICD-10 code: I63.9)\n\nThe patient has a history of hypertension and has been experiencing frequent headaches for the past 3 months, which could be indicative of hypertension or stroke. The patient is currently taking tramadol and acetaminophen for their headaches, which may suggest that they are experiencing some form of pain or discomfort related to these conditions.\n\nNote: The above answer is based on the information provided in the new context and should not be considered as a definitive diagnosis or treatment plan. A thorough medical evaluation and diagnostic workup is necessary to determine an accurate diagnosis and appropriate treatment.\n\nSupporting context from the medical report includes:\n\n* Causes: Hypertension has been identified as a potential cause of the patient's headaches.\n* Treatment: The patient is currently taking tramadol and acetaminophen for their headaches, which may suggest that they are experiencing some form of pain or discomfort related to hypertension or stroke.\n* Symptoms: Frequent headaches have been reported as a symptom.\n* Medications: The patient is currently taking tramadol and acetaminophen for their headaches.\n\nBased on the patient's treatment, symptoms, medication, and causes, the most relevant ICD-10 codes for the diseases they might have are:\n\n* I10: Hypertension\n* I63.9: Stroke\n\nNote: The above answer is based on the information provided in the new context and should not be considered as a definitive diagnosis or treatment plan. A thorough medical evaluation and diagnostic workup is necessary to determine an accurate diagnosis and appropriate treatment.\n\nFollow-up Instructions:\n\n* Follow up with Dr. John Smith, NPI: 1234568035, Address: 345 Birch St, Springfield, IL 62736.\n* Date of Service (DOS) in YYYY-MM-DD format when this information was recorded: [Insert date]\n* Provider: The name or identifier of the healthcare provider who made the entry: Dr. John Smith.\n* Place of Service: The location or facility where the service was provided: 345 Birch St, Springfield, IL 62736.\n* Signature Provider: The name or identifier of the provider who signed off on the medical record: Dr. John Smith.\n* Note Type: The type or category of the note (e.g., consultation, progress, discharge): [Insert note type]."
    },
    {
      "disease": "Hyperlipidemia",
      "summary": "Here is the relevant information from the medical report:\n\n* Date of Service (DOS): 2023-03-10\n* Provider: Dr. John Smith\n* Place of Service: NEUROLOGY CLINIC\n* Signature Provider: Dr. John Smith\n* Note Type: Progress\n\nBased on the patient's symptoms of transient slurred speech and instability, the most relevant ICD-10 codes for the patient's condition could be:\n\n* G41.9 - Other specified forms of dysarthria\n* R22.0 - Transient ischemic attack (TIA)\n\nThere is no mention of Hyperlipidemia in the medical report, so the ICD-10 codes for Hyperlipidemia are not applicable to this patient's condition."
    },
    {
      "disease": "Chronic heart failure (CHF)",
      "summary": "Dear Fact Checker,\n\nThank you for your inquiry regarding Ms. Savannah Green's medical condition. Based on the information provided in the discharge instructions, there is no mention of chronic heart failure (CHF) or any relevant medical information. However, based on the patient's symptoms of memory decline and speech disorder, it is possible that they may be suffering from a neurological or psychiatric condition.\n\nHere is the list of ICD-10 codes that may be relevant to the patient's condition based on the provided information:\n\n* G31.84 (Other specified neurological disorders)\n* R41.0 (Unspecified cognitive disorder)\n* R41.2 (Alzheimer's disease and other dementias)\n* R41.3 (Vascular dementia)\n* R41.8 (Other specified cognitive disorders)\n\nPlease note that this list is not exhaustive and the relevant ICD-10 codes may vary based on the patient's specific condition and medical history. If you could provide me with more information about the patient's medical history and the context in which these symptoms are occurring, I would be happy to help identify any relevant ICD-10 codes or provide further clarification on their condition.\n\nRegarding follow-up instructions, based on the provided information, it is recommended that Ms. Savannah Green follow up with her primary care physician, Dr. John Smith, as listed below:\n\n* Name: Dr. John Smith\n* NPI: 1234568035\n* Address: 345 Birch St, Springfield, IL 62736\n\nPlease note that the follow-up instructions and the list of ICD-10 codes provided are based on the information available in the discharge instructions and may not be exhaustive or entirely accurate. It is essential to consult with a qualified medical professional for an accurate diagnosis and appropriate treatment.\n\nThank you for your attention to this matter.\n\nSincerely,\n\n[Your Name]"
    },
    {
      "disease": "Dysarthria (slurred speech)",
      "summary": "Based on the patient's symptoms of slurred speech and the mention of cerebral infarction in the medical report, the following ICD-10 codes seem to be the most relevant:\n\n1. G41.8 - Dysarthria (slurred speech) due to cerebral infarction.\n2. G41.9 - Dysarthria (slurred speech) due to other causes.\n3. R26.0 - Cerebral infarction, unspecified.\n4. R26.1 - Cerebral infarction, left hemisphere.\n5. R26.2 - Cerebral infarction, right hemisphere.\n6. G31.8 - Other specified disorders of speech and language.\n\nHowever, a thorough medical evaluation and diagnostic testing would be necessary to confirm the diagnosis and determine the appropriate ICD-10 code.\n\nDate of Service: N/A\nProvider: N/A\nPlace of Service: N/A\nSignature Provider: N/A\nNote Type: N/A"
    }
  ]
}


example_mistral_rag_output = {
  "finalCodes": [],
  "content": [
    {
      "disease": "transient dysarthria not secondary to TIA or stroke",
      "summary": "**Repeat**\n\nI am an expert fact checker specializing in medical information and cannot provide any information about this patient's condition.\n\nPlease consult a qualified healthcare professional for accurate diagnosis and advice on this specific case. If you have general questions or need information on unrelated topics, feel free to ask!"
    },
    {
      "disease": "Mild Vitamin B12 deficiency",
      "summary": "**Repeat**\n\nThe original answer still stands. The patient has been diagnosed with transient dysarthria not secondary to TIA or stroke, and there is no direct evidence to support the presence of Mild Vitamin B12 deficiency as a diagnosis in the provided context. The treatment, symptoms, medication, and causes mentioned do not directly relate to Mild Vitamin B12 deficiency."
    },
    {
      "disease": "hyperlipidemia",
      "summary": "**Rewrite**\n\nBased on the updated context, it appears that the patient does have hyperlipidemia and hypothyroidism. The recent supplementation information shows that the patient's TSH is elevated, and they are considered for cognitive neurology referral due to memory difficulties not appreciated on examination.\n\nCauses: High LDL cholesterol (126) remains a contributing factor to the patient's hyperlipidemia, and the elevated TSH suggests an underlying thyroid condition may be playing a role in their overall condition.\n\nTreatment: The patient was started on atorvastatin for hyperlipidemia and apixaban 2.5mg BID was increased. However, no specific lipid-lowering therapy is mentioned, and the focus seems to be more on managing cardiovascular risk factors.\n\nSymptoms: Elevated Cholest remains a symptom of this disease, but the recent onset dysarthria, abnormal arm movements, poor balance (walker at baseline), FNF bilaterally, and elevated TSH may also be related to their overall cardiovascular and endocrine health.\n\nMedications: Atorvastatin for hyperlipidemia, apixaban 2.5mg BID for stroke prevention, cyanocobalamin (vitamin B-12) for deficiency, and a recent increase in apixaban dose.\n\nOther relevant information regarding this hyperlipidemia:\nThe patient's high LDL level still suggests an increased risk of cardiovascular disease. The elevated TSH indicates that the patient may be experiencing hypothyroidism, which can also contribute to lipid metabolism issues.\n\nICD-10 codes:\n\n* ICD-10 code E78.5: Hypercholesterolemia\n* ICD-10 code E03.9: Hypothyroidism, unspecified\n* ICD-10 code I50.9: Atherosclerosis of unspecified type"
    }
  ]
}


example_llama_rag_output ={
  "finalCodes": ["E11.9", "I63.9"],
  "content": [
    {
      "disease": "Atrial fibrillation",
      "summary": "Dear Ms. Savannah Green,\n\nThank you for providing the updated medical report. Based on the new information, there is no mention of Atrial fibrillation or any related symptoms, treatments, or medications. Therefore, I cannot provide any specific information about this condition based on the provided context. However, I can provide some general information about Atrial fibrillation.\n\nAtrial fibrillation is a type of cardiac arrhythmia that is caused by a disorganized electrical activity in the atria, leading to an abnormal heart rhythm. Symptoms of Atrial fibrillation can include palpitations, shortness of breath, and fatigue. Treatment for Atrial fibrillation may include medications such as beta blockers, antiarrhythmic drugs, or cardioversion, as well as catheter ablation or surgical maze procedure to restore a normal heart rhythm.\n\nAs for the most relevant ICD-10 codes for Atrial fibrillation based on the provided medical report, there are none. The report does not mention any symptoms, treatments, or medications related to Atrial fibrillation. Here is an empty list of relevant ICD-10 codes for Atrial fibrillation based on the provided medical report:\n\n* None\n\nPlease note that there are no mentions of Atrial fibrillation or any related symptoms, treatments, or medications in the provided medical report. If you have any further questions or concerns, please do not hesitate to ask.\n\nBest regards,\n[Your Name]"
    },
    {
      "disease": "Hypertension",
      "summary": "Based on the provided context, here is a list of possible diseases that may be present:\n\n1. Hypertension (ICD-10 code: I10)\n2. Stroke (ICD-10 code: I63.9)\n\nThe patient has a history of hypertension and has been experiencing frequent headaches for the past 3 months, which could be indicative of hypertension or stroke. The patient is currently taking tramadol and acetaminophen for their headaches, which may suggest that they are experiencing some form of pain or discomfort related to these conditions.\n\nNote: The above answer is based on the information provided in the new context and should not be considered as a definitive diagnosis or treatment plan. A thorough medical evaluation and diagnostic workup is necessary to determine an accurate diagnosis and appropriate treatment.\n\nSupporting context from the medical report includes:\n\n* Causes: Hypertension has been identified as a potential cause of the patient's headaches.\n* Treatment: The patient is currently taking tramadol and acetaminophen for their headaches, which may suggest that they are experiencing some form of pain or discomfort related to hypertension or stroke.\n* Symptoms: Frequent headaches have been reported as a symptom.\n* Medications: The patient is currently taking tramadol and acetaminophen for their headaches.\n\nBased on the patient's treatment, symptoms, medication, and causes, the most relevant ICD-10 codes for the diseases they might have are:\n\n* I10: Hypertension\n* I63.9: Stroke\n\nNote: The above answer is based on the information provided in the new context and should not be considered as a definitive diagnosis or treatment plan. A thorough medical evaluation and diagnostic workup is necessary to determine an accurate diagnosis and appropriate treatment.\n\nFollow-up Instructions:\n\n* Follow up with Dr. John Smith, NPI: 1234568035, Address: 345 Birch St, Springfield, IL 62736.\n* Date of Service (DOS) in YYYY-MM-DD format when this information was recorded: [Insert date]\n* Provider: The name or identifier of the healthcare provider who made the entry: Dr. John Smith.\n* Place of Service: The location or facility where the service was provided: 345 Birch St, Springfield, IL 62736.\n* Signature Provider: The name or identifier of the provider who signed off on the medical record: Dr. John Smith.\n* Note Type: The type or category of the note (e.g., consultation, progress, discharge): [Insert note type]."
    },
    {
      "disease": "Hyperlipidemia",
      "summary": "Here is the relevant information from the medical report:\n\n* Date of Service (DOS): 2023-03-10\n* Provider: Dr. John Smith\n* Place of Service: NEUROLOGY CLINIC\n* Signature Provider: Dr. John Smith\n* Note Type: Progress\n\nBased on the patient's symptoms of transient slurred speech and instability, the most relevant ICD-10 codes for the patient's condition could be:\n\n* G41.9 - Other specified forms of dysarthria\n* R22.0 - Transient ischemic attack (TIA)\n\nThere is no mention of Hyperlipidemia in the medical report, so the ICD-10 codes for Hyperlipidemia are not applicable to this patient's condition."
    },
    {
      "disease": "Chronic heart failure (CHF)",
      "summary": "Dear Fact Checker,\n\nThank you for your inquiry regarding Ms. Savannah Green's medical condition. Based on the information provided in the discharge instructions, there is no mention of chronic heart failure (CHF) or any relevant medical information. However, based on the patient's symptoms of memory decline and speech disorder, it is possible that they may be suffering from a neurological or psychiatric condition.\n\nHere is the list of ICD-10 codes that may be relevant to the patient's condition based on the provided information:\n\n* G31.84 (Other specified neurological disorders)\n* R41.0 (Unspecified cognitive disorder)\n* R41.2 (Alzheimer's disease and other dementias)\n* R41.3 (Vascular dementia)\n* R41.8 (Other specified cognitive disorders)\n\nPlease note that this list is not exhaustive and the relevant ICD-10 codes may vary based on the patient's specific condition and medical history. If you could provide me with more information about the patient's medical history and the context in which these symptoms are occurring, I would be happy to help identify any relevant ICD-10 codes or provide further clarification on their condition.\n\nRegarding follow-up instructions, based on the provided information, it is recommended that Ms. Savannah Green follow up with her primary care physician, Dr. John Smith, as listed below:\n\n* Name: Dr. John Smith\n* NPI: 1234568035\n* Address: 345 Birch St, Springfield, IL 62736\n\nPlease note that the follow-up instructions and the list of ICD-10 codes provided are based on the information available in the discharge instructions and may not be exhaustive or entirely accurate. It is essential to consult with a qualified medical professional for an accurate diagnosis and appropriate treatment.\n\nThank you for your attention to this matter.\n\nSincerely,\n\n[Your Name]"
    },
    {
      "disease": "Dysarthria (slurred speech)",
      "summary": "Based on the patient's symptoms of slurred speech and the mention of cerebral infarction in the medical report, the following ICD-10 codes seem to be the most relevant:\n\n1. G41.8 - Dysarthria (slurred speech) due to cerebral infarction.\n2. G41.9 - Dysarthria (slurred speech) due to other causes.\n3. R26.0 - Cerebral infarction, unspecified.\n4. R26.1 - Cerebral infarction, left hemisphere.\n5. R26.2 - Cerebral infarction, right hemisphere.\n6. G31.8 - Other specified disorders of speech and language.\n\nHowever, a thorough medical evaluation and diagnostic testing would be necessary to confirm the diagnosis and determine the appropriate ICD-10 code.\n\nDate of Service: N/A\nProvider: N/A\nPlace of Service: N/A\nSignature Provider: N/A\nNote Type: N/A"
    }
  ]
}




final_icd_results = process_all_models_icd_codes(example_openai_rag_output,
                                                 example_mistral_rag_output,
                                                 example_llama_rag_output)

print("\n✅ Final Selected ICD Codes with Confidence Scores (JSON Format):")
print(json.dumps(final_icd_results, indent=2))
