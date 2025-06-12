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
mistral_agent = GroqAgent(name="Groq_Mistral_Agent", model="llama-3.3-70b-versatile")
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
    if response.startswith(""):
        parts = response.split("\n")
        if parts[0].strip().startswith(""):
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
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

def extract_valid_evidence(summary: str) -> List[str]:
    """
    Extract valid evidence sentences from the summary.
    Only sentences with more than 5 words are included.
    """
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    valid_sentences = [sentence.strip() for sentence in sentences if len(sentence.split()) > 5]
    return valid_sentences if valid_sentences else ["No evidence provided"]

# --- Per-Model Processing Function --- 
def process_model_icd_codes(rag_output: Dict[str, Any], agent, num_codes: int = 5) -> List[Dict[str, Any]]:
    if "dxCodes" in rag_output:
        initial_codes = rag_output.get("dxCodes", [])[:num_codes]
    else:
        initial_codes = rag_output.get("finalCodes", [])[:num_codes]

    if "summaryInfo" in rag_output:
        joint_summary = "\n".join([entry.get("text", "") for entry in rag_output.get("summaryInfo", [])])
    else:
        joint_summary = "\n".join([entry.get("summary", "") for entry in rag_output.get("content", [])])

    descriptions = get_icd_descriptions(initial_codes)
    validated_codes, _ = validate_icd_codes(agent, initial_codes, descriptions, joint_summary, max_retries=3)
    final_codes = validated_codes[:num_codes]
    final_results = []
    for code in final_codes:
        desc = get_icd_descriptions([code])[code]
        confidence_result = get_confidence_and_evidence(agent, code, desc, joint_summary)
        # Use valid evidences extracted solely from the summary text.
        valid_evidence = extract_valid_evidence(joint_summary)
        final_results.append({
            "code": code,
            "description": desc,
            "confidence_score": confidence_result["score"],
            "evidence": valid_evidence
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
        "OpenAI": {"ICD10Codes": openai_results},
        "Mistral": {"ICD10Codes": mistral_results},
        "LLaMA": {"ICD10Codes": llama_results}
    }
    return final_results

# --- Conversion to ICD-10 Schema Output ---
def convert_result_to_icd10_schema(result: Dict[str, Any]) -> Dict[str, Any]:
    # Use the first evidence sentence truncated to 10 words for 'Text'
    evidence_list = result.get("evidence", [])
    if evidence_list and isinstance(evidence_list, list):
        first_evidence = evidence_list[0]
        text_words = first_evidence.split()[:10]
        text = " ".join(text_words)
    else:
        text = "No text provided"
    
    # Map fields using defaults where information is not available.
    schema_obj = {
        "Text": text,
        "disease": result.get("description", "Unknown disease"),
        "Category": "General",  # default value
        "Type": "Default",      # default value
        "Score": result.get("confidence_score", 50),
        "Attributes": [
            {
                "type": "evidence",
                "score": result.get("confidence_score", 50),
                "relationshipScore": 50,  # default value
                "text": ev
            } for ev in result.get("evidence", ["No evidence provided"])
        ],
        "Traits": [
            {
                "Name": "default",
                "Score": result.get("confidence_score", 50)
            }
        ],
        "ICD10CMConcepts": [
            {
                "Description": result.get("description", "No description"),
                "Code": result.get("code", "Unknown"),
                "hccCode": "24",
                "Score": result.get("confidence_score", 50)
            }
        ],
        "DOS": "01-01-2020",          # default date in MM-DD-YYYY
        "Provider": "Unknown Provider",
        "PlaceOfService": "Unknown",
        "SignatureProvider": "Unknown",
        "NoteType": "Unknown",
        "PageNumbers": []
    }
    return schema_obj

def convert_to_icd10_schema(results: Dict[str, Any]) -> Dict[str, Any]:
    new_results = {}
    for model, model_data in results.items():
        new_results[model] = {"ICD10Codes": []}
        for entry in model_data.get("ICD10Codes", []):
            schema_entry = convert_result_to_icd10_schema(entry)
            new_results[model]["ICD10Codes"].append(schema_entry)
    return new_results

# --- Example Inputs ---
example_openai_ragoutput = {
  "chartId": "679d3e1b316d5bed313c187b",
  "MemberId": "10002348",
  "llm": "mistral",
  "dxCodes": [
    "C70.9",
    "I47.0",
  ],
  "previouslSubmittedCodes": [
    "C7931",
    "C3490",
    "G936",
    "G935"
  ],
  "summaryInfo": [
    {
      "disease": "Bradycardia",
      "text": " Based on the provided context, it appears that the patient has undergone surgery to remove a brain lesion, experienced some post-operative symptoms such as headaches and incisional pain, and is currently taking medications like Dexamethasone for intracranial swelling. However, Bradycardia isnt explicitly mentioned or diagnosed in the given context.\n\nIn terms of ICD-10 codes for the conditions that are present:\n\n- G45.9 - Cerebellar infarction, unspecified (if related to the cerebral lesion)\n- If small cell lung carcinoma is confirmed in the patients follow-up, the code would be C71.9 - Carcinoma of trachea, bronchus, and lung, unspecified (other and unspecified).\n\nRegarding the medical entries details such as Date of Service, Provider, Place of Service, Signature Provider, and Note Type, they werent included in the provided context. However, assuming the information is available, here are some examples for each category:\n\n- Date of Service (DOS): YYYY-MM-DD (e.g., 2023-03-15)\n- Provider: John Doe MD or Provider ID (e.g., DrJohnDoe, PID123456789)\n- Place of Service: Hospital Name, Clinic Name, or Facility ID (e.g., Mayo Clinic, Blue Cross Clinic, FAC01234)\n- Signature Provider: John Doe MD, APRN, or PA (e.g., DrJohnDoe, NursePractitioner, PhysicianAssistant)\n- Note Type: Consultation, Progress, Discharge, etc. (e.g., Consultation - Initial Evaluation, Progress - Follow-up, Discharge - Final Diagnosis)"
    },
    {
      "disease": "Bells Palsy",
      "text": " Based on the given context, there is no evidence suggesting that the patient has Bells Palsy. However, the patient was diagnosed with a brain tumor as indicated by the discharge diagnosis.\n\n- Causes: The cause for the brain tumor is not explicitly mentioned in the provided context.\n\n- Treatment: The patient received medications such as Valacyclovir, Prenisolone gtts, Acetaminophen, Bisacodyl, Dexamethasone, Docusate Sodium, Famotidine, Levothyroxine, Lisinopril, PrednisoLONE Acetate Ophth. Susp., and Vitamin D during their stay at the hospital. These medications were not specified as related to Bells Palsy but rather for managing symptoms associated with the brain tumor or other conditions.\n\n- Symptoms: The patient had symptoms such as urinary urgency and increased frequency, but there are no symptoms mentioned in the context that suggest Bells Palsy.\n\n- Medications: The patient takes several medications, but their purpose is not specified as related to Bells Palsy.\n\nAs for the ICD-10 codes, since there is no evidence of Bells Palsy in this case, no ICD-10 codes are applicable. The ICD-10 codes provided earlier are related to the potential cerebellar lesion and its treatment.\n\nThere is also no available information regarding the Date of Service, Provider, Place of Service, Signature Provider, or Note Type for each entry."
    },
    {
      "disease": "Urinary Urgency",
      "text": " Based on the provided context, there is no specific information regarding Urinary Urgency for this patient. However, I can provide information about the symptoms, treatment, and possible causes of the conditions mentioned in the medical report, as well as suggest some relevant ICD-10 codes for the diseases they might have.\n\n1. Symptoms (Urinary Urgency): There are no reported symptoms related to Urinary Urgency in the provided context.\n2. Treatment (Urinary Urgency): No treatment for Urinary Urgency was mentioned in the provided context.\n3. Causes (Urinary Urgency): Since there is no information about this issue, its not possible to determine the cause of the Urinary Urgency.\n4. Medications (Urinary Urgency): No medications were prescribed specifically for Urinary Urgency in the provided context.\n5. Symptoms (Bradycardia): The patient was asymptomatic, but her heart rate dipped to the 50s intermittently.\n6. Treatment (Bradycardia): The patient improved with fluids and administration of her levothyroxine.\n7. Causes (Bradycardia): The cause of Bradycardia was not specified in the provided context, but it could be related to her underlying condition or medication side effects.\n8. Medications (Bradycardia): Other medications the patient is taking include Acetaminophen, Bisacodyl, Dexamethasone, Docusate Sodium, Famotidine, Polyethylene Glycol, Senna, Levothyroxine, Lisinopril, Prednisolone Ophth. Susp., Valacyclovir, Vitamin D, and Glucose Meter supplies.\n9. Symptoms (Bells Palsy): Not mentioned in the provided context.\n10. Treatment (Bells Palsy): The patient was resumed on her home Valacyclovir and Prenisolone gtts.\n11. Causes (Bells Palsy): Bells palsy is typically caused by reactivation of a latent viral infection, such as Herpes Simplex or Varicella-Zoster virus.\n12. Medications (Bells Palsy): Other medications the patient is taking include Acetaminophen, Bisacodyl, Dexamethasone, Docusate Sodium, Famotidine, Polyethylene Glycol, Senna, Levothyroxine, Lisinopril, Prednisolone Ophth. Susp., Valacyclovir, Vitamin D, and Glucose Meter supplies.\n13. Symptoms (Dispo): The patient was discharged in stable condition with no reported symptoms.\n14. Treatment (Dispo): No specific treatment was mentioned for her discharge.\n15. Causes (Dispo): Not specified in the provided context.\n16. Medications (Dispo): The patient continues to take Acetaminophen, Bisacodyl, Docusate Sodium, Famotidine, Polyethylene Glycol, Senna, Levothyroxine, Lisinopril, Prednisolone Ophth. Susp., Valacyclovir, Vitamin D, and Glucose Meter supplies upon discharge.\n\nAs for the ICD-10 codes related to these conditions:\n\n- Bradycardia: I45.9 (Other specified conduction disorders) or I47.0 (Sick sinus syndrome) based on the patients heart rate and symptoms.\n- Bells Palsy: G51.0 (Bells palsy) if it was confirmed in the medical report, but since it is not explicitly mentioned, no code can be assigned with certainty.\n- Urinary Urgency: R30.90 (Urinary urgency not caused by neurogenic bladder, unspecified) as it was mentioned on POD 2, but there is no specific information to confirm this diagnosis.\n- Dispo (Discharge Diagnosis - Brain Tumor): C71.9 (Malignant neoplasm of brain and intracranial parts of the nervous system, unspecified) if the brain tumor was confirmed in the medical report. However, since the focus of the report is not on this issue, no code can be assigned with certainty.\n\nThe following information has been added to each section as requested:\n\n- Date of Service (DOS): The date of service information is not provided in the given context for each entry.\n- Provider: The name or identifier of the healthcare provider who made the entry is also not specified in the provided context.\n- Place of Service: The location or facility where the service was provided is missing from the report.\n- Signature Provider: There is no signature of the provider on the medical record in the given context.\n- Note Type: The type or category of the note (e.g., consultation, progress, discharge) is not specified for each entry."
    },
    {
      "disease": "Small Cell Lung Carcinoma",
      "text": "1. Causes: While the specific cause of the patients Small Cell Lung Carcinoma is not explicitly stated, its recognized that smoking is a significant risk factor for this type of lung cancer, as no other causative factors are mentioned in the provided context.\n\n2. Treatment (on 10-10-2023): The patient underwent surgical resection of her cerebellar lesion which was identified as Small Cell Lung Carcinoma; however, no additional treatment details specific to the lung cancer were given in the provided context.\n\n3. Symptoms: Although the patients symptoms associated with her Small Cell Lung Carcinoma were not explicitly detailed in the context, she was diagnosed based on the pathology of the resected brain lesion. Common symptoms include cough, chest pain, shortness of breath, and weight loss; however, during her hospital stay, she experienced urinary urgency and increased frequency on 10-12-2023 (POD 2).\n\n4. Medications: During her hospitalization, the patient received dexamethasone for mass effect and occasionally required sliding scale Insulin due to steroid-induced hyperglycemia while on Dexamethasone. No specific medication was mentioned for treating the Small Cell Lung Carcinoma in the discharge medications list.\n\n5. Other relevant information: The patient has a history of a cerebral aneurysm and related conditions such as a cerebellar lesion and hydrocephalus. Additionally, she experienced steroid-induced hyperglycemia during her hospital stay.\n\n6. ICD-10 codes (based on the information provided):\n   - C34.9: Small cell lung carcinoma (for the diagnosed lung cancer)\n   - G43.909: Other specified intracranial aneurysms, unspecified side (for the cerebral aneurysm)\n   - Z79.51: Other mental disorder due to known physical hazards (for anxiety related to her health condition)\n   - R30.0: Urinary urgency and frequency (for the symptoms experienced on POD 2).\n\n- Date of Service (DOS): The provided context does not contain specific dates for each entry.\n- Provider: The name or identifier of the healthcare provider who made the entry is not specified in the provided context.\n- Place of Service: The location or facility where the service was provided is also unspecified in the given context.\n- Signature Provider: There is no signature from a provider on the medical record as presented.\n- Note Type: The type or category of each note (e.g., consultation, progress, discharge) is not explicitly mentioned in the provided context."
    }
  ]
}

example_mistral_ragoutput = {
  "chartId": "67a250edc4a16e87f4865b54",
  "MemberId": "10002348",
  "llm": "llama3.1:8b-instruct-q4_0",
  "dxCodes": ["I67.0"],
  "previouslSubmittedCodes": ["G911", "C7931", "C3490", "G936", "G935"],
  "summaryInfo": [
    {
      "disease": "Bradycardia - She intermittently dipped to the 50s, however remained asymptomatic.",
      "text": "**Rewrite**\n\nThe patient has been experiencing asymptomatic bradycardia, with her heart rate intermittently dipping to the 50s. This condition was likely caused by a lack of symptoms or underlying cardiac conditions.\n\nTreatment for this condition included administration of fluids and levothyroxine, which helped improve her heart rate. The patient also remained asymptomatic during this time. Additionally, the patient's thyroid function may have been impacted due to the administration of levothyroxine.\n\nSymptoms that indicate this bradycardia include the patient's heart rate dipping to the 50s, although she remained asymptomatic. Other symptoms or conditions related to this bradycardia are not explicitly mentioned in the provided context.\n\nMedications taken by the patient to cure or manage this condition include levothyroxine, which was administered to help improve her heart rate and potentially address thyroid dysfunction.\n\nBased on the patient's treatment, symptoms, medication, and causes, some relevant ICD-10 codes for diseases they might have are:\n\n* R047 - Bradycardia (asymptomatic)\n\t+ DOS: 2023-10-10\n\t+ Provider: Springfield Hospital\n\t+ Place of Service: Inpatient unit\n\t+ Signature Provider: [Signature]\n\t+ Note Type: Progress note\n\n* E46.8 - Thyroid dysfunction, unspecified\n\t+ DOS: 2023-10-10\n\t+ Provider: Springfield Hospital\n\t+ Place of Service: Inpatient unit\n\t+ Signature Provider: [Signature]\n\t+ Note Type: Progress note"
    },
    {
      "disease": "Bell's palsy - The patient was resumed on her home Valacyclovir and Prenisolone gtts.",
      "text": "**Rewrite**\n\nThe patient does not have Bell's palsy. The pertinent results mention an asymptomatic bradycardia with heart rates intermittently dipping to the 50s, which remained asymptomatic after receiving fluids and levothyroxine.\n\nCauses: The patient's heart rate fluctuations are likely related to her underlying medical condition.\nTreatment: The patient received dexamethasone for mass effect.\nSymptoms: There are no facial strength or sensation issues related to cranial nerves V, VII.\nMedications: The patient is taking Valacyclovir and Prenisolone gtts. and Dexamethasone.\n\nRelevant ICD-10 codes for the actual condition could be:\nR007 - Asymptomatic bradycardia\nG047 - Cerebellar hypodensity\nI67.0 - Hydrocephalus\n\n**Date of Service:** Not provided.\n**Provider:** Unknown.\n**Place of Service:** Unknown.\n**Signature Provider:** Unknown.\n**Note Type:** Medical report."
    },
    {
      "disease": "Urinary retention",
      "text": "**Rewrite**\n\nThere is no indication of severe pain, swelling, redness or drainage from the incision site, fever greater than 101.5 degrees Fahrenheit, nausea and/or vomiting, extreme sleepiness and not being able to stay awake, severe headaches not relieved by pain relievers, seizures, new problems with your vision or ability to speak, weakness or changes in sensation in your face, arms, or leg in the provided medical report. However, one possible issue is related to sudden numbness or weakness in the face, arm, or leg, which could be a sign of stroke. \n\nCauses: None mentioned directly related to these symptoms.\nTreatment: Management included fluids and monitoring for bradycardia, Bell's palsy, and urinary symptoms.\nSymptoms: Complaints of severe headaches not relieved by pain relievers on POD 2.\nMedications: Valacyclovir and Prenisolone gtts for Bell's palsy, Levothyroxine administration helped her heart rate improvement in the context of bradycardia, and management included fluids.\nOther relevant information: Her asymptomatic bradycardia was managed with fluids and levothyroxine.\n\nRelevant ICD-10 codes based on the provided context:\n- G44.411 - Tension-type headache\n- H04.419 - Diplopia (double vision)\n- R42.8 - Other symptoms involving autonomic nervous system \n\n**Dates and Providers**\n- Date of Service (DOS): 20XX-XX-XX\n- Provider: Not specified in this report snippet.\n- Place of Service: ICU or POD1 (not detailed).\n- Signature Provider: Not provided in the given context.\n- Note Type: Progress note for the management of bradycardia, Bell's palsy, and urinary symptoms."
    }
  ]
}

example_llama_ragoutput = {
  "chartId": "67a250edc4a16e87f4865b54",
  "MemberId": "10002348",
  "llm": "llama3.1:8b-instruct-q4_0",
  "dxCodes": [
    "I67.0"
  ],
  "previouslSubmittedCodes": [
    "G911",
    "C7931",
    "C3490",
    "G936",
    "G935"
  ],
  "summaryInfo": [
    {
      "disease": "Bradycardia - She intermittently dipped to the 50s, however remained asymptomatic.",
      "text": "**Rewrite**\n\nThe patient has been experiencing asymptomatic bradycardia, with her heart rate intermittently dipping to the 50s. This condition was likely caused by a lack of symptoms or underlying cardiac conditions.\n\nTreatment for this condition included administration of fluids and levothyroxine, which helped improve her heart rate. The patient also remained asymptomatic during this time. Additionally, the patients thyroid function may have been impacted due to the administration of levothyroxine.\n\nSymptoms that indicate this bradycardia include the patients heart rate dipping to the 50s, although she remained asymptomatic. Other symptoms or conditions related to this bradycardia are not explicitly mentioned in the provided context.\n\nMedications taken by the patient to cure or manage this condition include levothyroxine, which was administered to help improve her heart rate and potentially address thyroid dysfunction.\n\nBased on the patients treatment, symptoms, medication, and causes, some relevant ICD-10 codes for diseases they might have are:\n\n* R047 - Bradycardia (asymptomatic)\n\t+ DOS: 2023-10-10\n\t+ Provider: Springfield Hospital\n\t+ Place of Service: Inpatient unit\n\t+ Signature Provider: [Signature]\n\t+ Note Type: Progress note\n\n* E46.8 - Thyroid dysfunction, unspecified\n\t+ DOS: 2023-10-10\n\t+ Provider: Springfield Hospital\n\t+ Place of Service: Inpatient unit\n\t+ Signature Provider: [Signature]\n\t+ Note Type: Progress note"
    },
    {
      "disease": "Bells palsy - The patient was resumed on her home Valacyclovir and Prenisolone gtts.",
      "text": "**Rewrite**\n\nThe patient does not have Bells palsy. The pertinent results mention an asymptomatic bradycardia with heart rates intermittently dipping to the 50s, which remained asymptomatic after receiving fluids and levothyroxine.\n\nCauses: The patients heart rate fluctuations are likely related to her underlying medical condition.\nTreatment: The patient received dexamethasone for mass effect.\nSymptoms: There are no facial strength or sensation issues related to cranial nerves V, VII.\nMedications: The patient is taking Valacyclovir and Prenisolone gtts. and Dexamethasone.\n\nRelevant ICD-10 codes for the actual condition could be:\nR007 - Asymptomatic bradycardia\nG047 - Cerebellar hypodensity\nI67.0 - Hydrocephalus\n\n**Date of Service:** Not provided.\n**Provider:** Unknown.\n**Place of Service:** Unknown.\n**Signature Provider:** Unknown.\n**Note Type:** Medical report."
    },
    {
      "disease": "Urinary retention",
      "text": "**Rewrite**\n\nThere is no indication of severe pain, swelling, redness or drainage from the incision site, fever greater than 101.5 degrees Fahrenheit, nausea and/or vomiting, extreme sleepiness and not being able to stay awake, severe headaches not relieved by pain relievers, seizures, new problems with your vision or ability to speak, weakness or changes in sensation in your face, arms, or leg in the provided medical report. However, one possible issue is related to sudden numbness or weakness in the face, arm, or leg, which could be a sign of stroke. \n\nCauses: None mentioned directly related to these symptoms.\nTreatment: Management included fluids and monitoring for bradycardia, Bells palsy, and urinary symptoms.\nSymptoms: Complaints of severe headaches not relieved by pain relievers on POD 2.\nMedications: Valacyclovir and Prenisolone gtts for Bells palsy, Levothyroxine administration helped her heart rate improvement in the context of bradycardia, and management included fluids.\nOther relevant information: Her asymptomatic bradycardia was managed with fluids and levothyroxine.\n\nRelevant ICD-10 codes based on the provided context:\n- G44.411 - Tension-type headache\n- H04.419 - Diplopia (double vision)\n- R42.8 - Other symptoms involving autonomic nervous system \n\n**Dates and Providers**\n- Date of Service (DOS): 20XX-XX-XX\n- Provider: Not specified in this report snippet.\n- Place of Service: ICU or POD1 (not detailed).\n- Signature Provider: Not provided in the given context.\n- Note Type: Progress note for the management of bradycardia, Bells palsy, and urinary symptoms."
    }
  ]
}

# --- Process the outputs from all three models ---
final_icd_results = process_all_models_icd_codes(example_openai_ragoutput, example_mistral_ragoutput, example_llama_ragoutput)

# --- Convert final results to the ICD-10 Schema format ---
final_schema_output = convert_to_icd10_schema(final_icd_results)

# --- Save final ICD-10 Schema output to a JSON file ---
output_filename = "icd_schema_output.json"
with open(output_filename, "w") as outfile:
    json.dump(final_schema_output, outfile, indent=2)
print(f"\n✅ Final Selected ICD Codes saved to '{output_filename}'")
