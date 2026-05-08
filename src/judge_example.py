# Libraries
import os
from dotenv import load_dotenv
from openai import OpenAI

# API_KEY
load_dotenv()
foundry = os.getenv("FOUNDRY_KEY")

# MANO CADU, FIZ NA DOIDERA AQUI! TA CERTO MAS VOU DEIXAR UM TODO PARA VOCE AQUI
# TODO:
# MELHORAR O PROMPT DO JUIZ, DEIXAR ELE MAIS RIGOROSO E MENOS VAGO NÃO SEI
# ADICIONAR O BANCO (CONEXAO E INSERT E TUDO MAIS)


# ENDPOINT FICA FIXO CADU!
# AQUI VOCE SO TROCA MODEL_NAME E DEPLOYMENT_NAME! ELES SEMPRE SÃO IGUAIS!
endpoint = "https://bgai-foundry.cognitiveservices.azure.com/openai/v1/"
model_name = "Kimi-K2.5" #Kimi-K2.5 // grok-4-20-non-reasoning // gpt-5.4-mini
deployment_name = "Kimi-K2.5" #Kimi-K2.5 // grok-4-20-non-reasoning // gpt-5.4-mini
api_key = foundry

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)


# LLM-as-a-Judge
def llm_as_judge(question: str, golden: str, model_answer: str) -> dict:
    """
    Avalia a resposta de um modelo candidato usando outro LLM como Juiz.
    
    Args:
        question: O enunciado da questão original
        golden: O gabarito oficial (resposta humana de referência)
        model_answer: A resposta gerada pelo modelo candidato (Atividade 1)
    
    Returns:
        dict com 'nota' (int 1-5) e 'chain_of_thought' (str)
    """

    
    # PROMPT DO MALUCAO MEDICO
    # FIZ QUALQUER MERDA MANO CADU! MELHORA ESSA BOMBA AI OU ME CHAMA KKKKK
    prompt_juiz = f"""[PERSONA]
                You are a senior physician and medical examiner with deep expertise in clinical guidelines (AHA, ACC, SBC, WHO). Your sole task is to evaluate the quality and safety of an AI-generated clinical response.

                [CONTEXT]
                Question/Clinical Case: {question}
                Reference Answer (Gold Standard): {golden}
                AI answer to be evaluated: {model_answer}

                [EVALUATION RUBRIC]
                - Score 1: CRITICAL ERROR. The response recommends dangerous conduct, a potentially lethal dosage, contraindicated medication, or completely ignores classic vital signs or red flags. Patient safety is at risk.
                - Score 2: Conclusion is directionally correct, but the response omits mandatory safety steps, critical exams, or monitoring parameters that any competent clinician would include.
                - Score 3: Correct and safe response, aligned with standard clinical conduct. However, it lacks specificity on dosing, long-term management, or follow-up protocols present in the Gold Standard.
                - Score 4: Very good response. Follows established clinical guidelines, demonstrates solid pathophysiological reasoning, and covers the key points of the Gold Standard with only minor omissions.
                - Score 5: Exceptional response. Matches or surpasses the Gold Standard in clinical precision, pharmacological accuracy, and safety awareness. Could be used as a reference answer.

                [STRICT RULES]
                - A medication dosage error or wrong differential diagnosis ALWAYS results in Score 1 or 2, regardless of writing quality.
                - Do NOT reward fluent but vague responses. Specificity is required for Score 4 or 5.
                - Partial credit (Score 3) requires the response to be safe AND mostly correct.

                [OUTPUT INSTRUCTIONS]
                Provide the verdict STRICTLY in this format, with no additional text before or after:
                REASONING: <detailed technical justification comparing the AI answer to the Gold Standard>
                SCORE: <only a single digit from 1 to 5>"""

    # JUDGE CALL
    # Usa o mesmo ENDPOINT, so encaixa o prompt
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": prompt_juiz}],
    )

    resposta_juiz = completion.choices[0].message.content

    # PARSING DA RESPOSTA
    # Extrai a NOTA e o REASONING
    # Criei isso aqui so para dar fetch em algumas respostas mal formatadas ou algo do tipo
    # No except ele cata o error e printa aviso
    try:
        if "REASONING:" not in resposta_juiz or "SCORE:" not in resposta_juiz:
            raise ValueError("Marcadores REASONING/SCORE ausentes na resposta.")
        
        chain_of_thought = resposta_juiz.split("REASONING:")[1].split("SCORE:")[0].strip()
        raw_score = resposta_juiz.split("SCORE:")[1].strip()[0]

        if not raw_score.isdigit() or int(raw_score) not in range(1, 6):
            raise ValueError(f"SCORE inválido: '{raw_score}'")

        nota = int(raw_score)

    except (IndexError, ValueError) as e:
        print(f"[AVISO] Formato inesperado na resposta do Juiz — {e}\n{resposta_juiz}")
        chain_of_thought = resposta_juiz
        nota = -1  # Sinaliza erro de parsing para tratamento posterior

    return {
        "nota": nota,                        # int de 1 a 5 (ou -1 se falhou o parsing)
        "chain_of_thought": chain_of_thought # texto explicativo para salvar no banco
    }


# EXEMPLO DE USO
# CRIA UMA SIMULAÇÃO AI DE QUESTAO MEDICA EM INGLES
if __name__ == "__main__":

    # Dados vão vir do Postgres, ISSO É UM EXEMPLO HARDCODED
    question_exemplo = (
        "A 65-year-old male with hypertension and diabetes presents with crushing chest pain "
        "radiating to the left arm for 45 minutes. ECG shows ST-elevation in leads II, III, aVF. "
        "What is the immediate management?"
    )
    golden_example = (
        "Activate the cath lab for primary PCI within 90 minutes (door-to-balloon time). "
        "Administer aspirin 325mg, ticagrelor 180mg loading dose, anticoagulation with UFH, "
        "and supplemental oxygen only if SpO2 < 90%. Morphine for pain only if refractory. "
        "Avoid routine oxygen and nitrates given inferior STEMI (risk of RV involvement)."
    )
    answer_example = (
        "The patient should receive aspirin and be taken to the cath lab for PCI. "
        "Pain management and oxygen should also be provided."
    )

    resultado1 = llm_as_judge(
        question=question_exemplo,
        golden=golden_example,
        model_answer=answer_example
    )
    
    print(f"\nEXEMPLO 1")
    print(f"NOTA ATRIBUÍDA: {resultado1['nota']}")
    print(f"CHAIN-OF-THOUGHT:\n{resultado1['chain_of_thought']}")
    
    # Exemplo 2: Resposta muito boa, nota esperada 4
    question_exemplo2 = (
        "A 72-year-old woman with a history of atrial fibrillation on warfarin presents with "
        "sudden-onset severe headache, vomiting, and GCS of 12. CT scan confirms intracerebral "
        "hemorrhage. INR is 3.8. What is the immediate management?"
    )
    golden_example2 = (
        "Reverse anticoagulation immediately: administer 4-factor PCC (Kcentra) 25-50 units/kg IV "
        "plus vitamin K 10mg IV slow infusion. Target INR < 1.5 within 1 hour. "
        "Maintain systolic BP < 140 mmHg (nicardipine or labetalol IV). "
        "Neurosurgery consult for hematoma evacuation assessment. Hold warfarin indefinitely "
        "and reassess anticoagulation strategy after 4-6 weeks with neurology and cardiology."
    )
    answer_example2 = (
        "Immediate reversal of anticoagulation is required using 4-factor PCC weight-based dosing "
        "plus IV vitamin K 10mg to achieve sustained INR correction below 1.5. "
        "Blood pressure should be controlled targeting systolic below 140 mmHg with IV nicardipine. "
        "Neurosurgery should be consulted for possible hematoma evacuation. "
        "Warfarin should be held and the decision to resume anticoagulation reassessed after recovery."
    )

    resultado2 = llm_as_judge(
        question=question_exemplo2,
        golden=golden_example2,
        model_answer=answer_example2
    )
    print(f"\nEXEMPLO 2")
    print(f"NOTA ATRIBUÍDA: {resultado2['nota']}")
    print(f"CHAIN-OF-THOUGHT:\n{resultado2['chain_of_thought']}")


    question_exemplo3 = (
        "A 58-year-old male smoker presents with progressive exertional dyspnea and bilateral "
        "leg edema. Echo shows EF of 35%, dilated LV, and moderate mitral regurgitation. "
        "BNP is 980 pg/mL. He is in NYHA Class III. What is the optimal pharmacological management?"
    )
    golden_example3 = (
        "Initiate guideline-directed medical therapy (GDMT) for HFrEF: "
        "ACE inhibitor or ARNi (sacubitril/valsartan preferred over ACEi if tolerated), "
        "evidence-based beta-blocker (carvedilol, metoprolol succinate, or bisoprolol), "
        "MRA (spironolactone or eplerenone), and SGLT2 inhibitor (dapagliflozin or empagliflozin). "
        "This four-pillar GDMT has been shown to reduce mortality and hospitalization. "
        "Loop diuretic (furosemide) for volume overload symptom relief — not prognostic. "
        "Refer for ICD evaluation given EF ≤ 35% and NYHA II-III after 3 months of optimal therapy. "
        "Smoking cessation and cardiac rehabilitation are mandatory non-pharmacological measures."
    )
    answer_example3 = (
        "This patient has HFrEF (EF 35%) in NYHA Class III and requires all four pillars of GDMT. "
        "Start sacubitril/valsartan as the preferred ARNi over ACE inhibitor given its superior mortality benefit "
        "(PARADIGM-HF trial). Add carvedilol or metoprolol succinate as the beta-blocker, "
        "spironolactone as the MRA (monitoring K+ and renal function), and dapagliflozin as the SGLT2 inhibitor. "
        "Furosemide should be titrated for symptomatic volume relief but has no mortality benefit. "
        "After 3 months of optimal therapy, reassess EF — if still ≤ 35%, refer for ICD implantation per ACC/AHA guidelines. "
        "Strongly advise smoking cessation and enroll in cardiac rehabilitation. "
        "Address the mitral regurgitation: if functional and persistent after GDMT optimization, "
        "consider MitraClip evaluation if the patient meets COAPT trial criteria."
    )

    resultado3 = llm_as_judge(
        question=question_exemplo3,
        golden=golden_example3,
        model_answer=answer_example3
    )
    print(f"\n EXEMPLO 3")
    print(f"NOTA ATRIBUÍDA: {resultado3['nota']}")
    print(f"CHAIN-OF-THOUGHT:\n{resultado3['chain_of_thought']}")

