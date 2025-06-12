"""
This module contains prompt templates for ICD code validation, alternative suggestion, 
and confidence scoring. Each template has been expanded to approximately 500 words 
to follow a detailed “perfect prompt” mechanism.
"""

VALIDATION_PROMPT_TEMPLATE = """
Dear Esteemed Medical Coding Expert,

We are conducting an in-depth evaluation of ICD-10 coding accuracy based on a comprehensive patient clinical summary. Your expert role is critical in ensuring that each ICD-10 code not only adheres to clinical standards but also captures the precise nature of the patient's condition. In the task ahead, you are provided with the ICD-10 code: {ICD_CODE} and its official description: {DESCRIPTION}. You are also given a detailed clinical summary: {SUMMARY}. Your mission is to thoroughly review the clinical information and determine whether the code is a perfect match for the patient's clinical picture.

Please consider all relevant factors including the primary diagnosis, any comorbidities, treatment details, and documented symptoms. If you believe that the code fully corresponds to the clinical scenario, kindly respond with the single word “CONFIRMED”. However, if you find discrepancies or a better alternative exists, please respond by providing an alternative ICD-10 code along with a succinct explanation. Your explanation must include 2–3 concise evidence phrases (each no longer than 20 words) extracted directly from the clinical summary that justify your suggestion.

Your analysis should be comprehensive and include a careful cross-reference with established coding guidelines. Consider the following:
1. Examine the full scope of the patient’s symptoms, history, and diagnostic tests.
2. Identify any nuances in the clinical summary that might affect the accuracy of the given code.
3. Provide evidence phrases that directly link specific clinical details to your coding decision.
4. If suggesting an alternative, ensure that your reasoning is clear and justified by the evidence.

In addition to the above, please reflect on the following:
- The impact of accurate coding on patient care and administrative processes.
- The importance of precision in clinical documentation.
- How variations in clinical presentation might require nuanced coding adjustments.
- The need to adhere to both national and international coding standards.

This detailed and explicit instruction is designed to help you extract maximum insight and clarity from your review. Your response will play a pivotal role in ensuring high-quality healthcare documentation and will serve as an essential reference for ongoing training and quality improvement initiatives. We trust that your comprehensive analysis will capture the full complexity of the patient’s situation.

Thank you for your careful analysis and unwavering commitment to excellence in medical coding. Your insights are invaluable, and we look forward to your response.
"""

ALTERNATIVE_SUGGESTION_PROMPT_TEMPLATE = """
Dear Expert Medical Coder,

Following your initial review of the ICD-10 codes, we request your refined recommendations for any codes that did not fully meet the accuracy criteria. You are provided with a list of previously submitted codes: {PREVIOUS_CODES} along with the detailed clinical summary: {SUMMARY}. Your task is to re-evaluate these codes and suggest alternatives that more accurately capture the clinical scenario.

Please provide your refined suggestions strictly in JSON format as shown below:
{
  "finalCodes": ["code1", "code2", "code3", "code4", "code5"]
}

In your re-evaluation, consider the following:
1. Thoroughly analyze the clinical summary, noting key symptoms, diagnostic findings, and treatment details.
2. Identify any discrepancies between the clinical information and the previously suggested ICD-10 codes.
3. Propose alternative codes that align more closely with the clinical evidence.
4. Ensure that each suggestion is based on clear evidence extracted from the summary, with direct reference to specific clinical details.
5. Avoid extraneous commentary—your response should contain only the JSON structure with the refined ICD codes.

Furthermore, please take into account:
- The distinction between primary and secondary diagnoses.
- How comorbidities or atypical presentations might necessitate alternative coding.
- The importance of following current coding guidelines and best practices.
- The critical role of precise coding in effective patient management and healthcare reporting.

Your refined recommendations are essential in ensuring the integrity of our coding process. We value your expertise and detailed insights in this complex evaluation.

Thank you for your diligence and for providing clear, evidence-based alternative ICD-10 codes.
"""

CONFIDENCE_PROMPT_TEMPLATE = """
Dear Esteemed Medical Coding Expert,

We require your expert evaluation to assign a confidence score to the provided ICD-10 code in relation to the patient’s clinical summary. You are given the ICD-10 code: {ICD_CODE} along with its official description: {DESCRIPTION} and the clinical summary: {SUMMARY}. Your task is to analyze the degree to which the code aligns with the clinical details and to express your evaluation in a strict JSON format with the following keys:
- "score": an integer between 0 and 100 (with scores between 90 and 100 indicating near-perfect alignment, 50 to 89 moderate alignment, and 0 to 49 indicating poor alignment).
- "evidence": a list of 2–3 concise evidence phrases (each no longer than 20 words) that capture the key clinical elements justifying your score.

Please follow these instructions meticulously:
1. Read the entire clinical summary, paying close attention to patient history, diagnostic findings, symptoms, and treatment interventions.
2. Evaluate whether the ICD-10 code and its corresponding description precisely match the clinical scenario.
3. Based on your evaluation, assign a numerical score that reflects the accuracy of the code.
4. Identify specific evidence from the clinical summary that supports your scoring decision. Your evidence phrases must be clear, directly linked to the clinical details, and should not exceed 20 words each.
5. Format your response strictly as a JSON object with only the two required keys ("score" and "evidence") and no additional commentary or extraneous text.

This request for a detailed, evidence-based scoring mechanism is crucial for guiding further quality assurance measures and coding improvements. Your methodical approach and clear justification will provide a strong foundation for ongoing refinement of our medical coding practices.

Thank you for your expertise and for providing a thorough, evidence-based response.
"""
