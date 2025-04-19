import json
import evaluate
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 모델과 토크나이저 로드
model_path = "./flan_tuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 평가용 데이터 불러오기 (Validation set이 있다면 그걸 사용)
with open("disaster_instruction_dataset_3000.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    
samples = raw_data[:50] # 일부 샘플로 평가 진행

# 평가 준비
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

predictions = []
references = []

print("\n[Step 1: 기본 입력 평가 - Prompt Engineering 없이]")
for sample in samples:
    input_text = f"Instruction: {sample['instruction']}\nInput: {sample['input']}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    
    output = model.generate(input_ids, max_length=128)
    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    
    predictions.append(prediction)
    references.append(sample["output"])
    
# ROUGE 계산
rouge_results = rouge.compute(predictions=predictions, references=references)

# BLEU 계산
bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

# BERTScore 계산
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")

# 결과출력
print("\n평가 지표 결과 요약:")
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"BLEU: {bleu_results['bleu']:.4f}")
print(f"BERTScore (F1): {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}")

# 결과 표 (논문용 텍스트 버전)
print("\n\n 논문용 성능 비교표:")
print("| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | BERTScore (F1) |")
print("|-------|----------|----------|----------|-------|-----------------|")
print(f"| FLAN-T5 튜닝 | {rouge_results['rouge1']:.4f} | {rouge_results['rouge2']:.4f} | {rouge_results['rougeL']:.4f} | {bleu_results['bleu']:.4f} | {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f} |")

# 논문용 성능 비교 분석 문단 생성
print("\n 논문용 분석 문단:")
print(f"본 연구에서 Instruction Tuning을 적용한 FLAN-T5 모델은 ROUGE-1 {rouge_results['rouge1']:.4f}, ROUGE-2 {rouge_results['rouge2']:.4f}, ROUGE-L {rouge_results['rougeL']:.4f}의 결과를 보였다. BLEU 점수는 {bleu_results['bleu']:.4f}, BERTScore(F1)는 {sum(bertscore_results['f1'])/len(bertscore_results['f1']):.4f}로 나타나, 전반적으로 모델이 재난 상황에 대한 요약 및 경고 문구를 일정 수준으로 생성할 수 있음을 확인했다. 다만, low-resource 학습 환경 및 instruction 표현의 다양성 부족으로 인해 일부 지표에서 낮은 성능을 보였으며, 향후 데이터 다양성과 학습 샘플 확대를 통해 성능 개선이 가능할 것으로 판단된다.")

# CSV로 저장
pd.DataFrame(results).to_csv("template_eval_results.csv", index=False)
print("\n평가 결과가 template_eval_results.csv 파일로 저장되었습니다.")
