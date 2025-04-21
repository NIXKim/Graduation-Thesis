import json
import openai
from time import sleep

openai.api_key = "your_api_key"

    # 평가용 데이터셋 생성 프롬프트
def generate_eval_dataset(batch_size=30):
    prompt = """
    너는 재난 대응 AI 모델의 성능을 평가하기 위한 instruction tuning 평가 데이터셋을 생성하는 AI야.

    다음 조건을 모두 반영하여 300개의 JSON 항목으로 구성된 평가용 데이터셋을 생성해줘:

    조건:
    1. 각 항목은 instruction, input, output 3개의 필드로 구성한다.
    2. instruction은 시민에게 행동 지침, 경고, 주의사항 등을 요구하는 다양한 형태로 작성하되, 학습 데이터와 표현이 겹치지 않도록 새로운 방식으로 구성한다.
    3. input은 실제 재난 상황을 설명하는 문장으로 구성하되, 뉴스 기사, 공공 경고, SNS 메시지 스타일을 포함하며, 주제는 학습 데이터에 없는 새로운 상황을 추가로 포함한다. (예: 도로 침수, 우박, 항공기 결항 등)
    4. output은 시민이 따라야 할 구체적이고 명확한 대응 행동을 설명하며, 간결하고 직접적인 문장으로 작성한다.
    5. 한글과 영어 항목을 번갈아 포함하며, 가능한 다양한 재난 유형을 포괄한다.
    6. 내용은 학습 데이터와 유사하지 않도록 하되, 같은 형식을 따르도록 한다.

    출력은 아래와 같은 JSON 리스트 형식으로 구성하되, 설명 없이 순수 JSON만 출력해줘:

    [
        {
            "instruction": "주어진 재난 상황에 대해 시민이 따라야 할 대응 지침을 제공하세요.",
            "input": "서울 동작구에서 도로가 침수되어 차량 통행이 어렵다는 신고가 접수되었습니다.",
            "output": "침수 지역으로의 차량 이동을 피하고, 고지대로 우회하세요."
        },
        ...
    ]
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        generated = response.choices[0].message.content
        print("응답 길이:", len(generated))
        
        try:
            dataset = json.loads(generated)
            return dataset
        except json.JSONDecodeError as je:
            print(f"JSON 파싱 에러: {je}")
            return []
    
    except Exception as e:
        print(f"OpenAI 요청 중 에러 발생: {e}")
        return []
    
    
def generate_full_dataset(target_total=300, batch_size=30):
    all_data = []
    for i in range(target_total // batch_size):
        print(f"{i+1}/{target_total // batch_size} - 데이터 생성 중...")
        batch = generate_eval_dataset(batch_size)
        if batch:
            all_data.extend(batch)
        else:
            print(f"{i+1}번째 배치 실패.")
        sleep(1.5)
    return all_data

def remove_duplicates(data):
    seen = set()
    unique = []
    for item in data:
        key = (item.get("instruction"), item.get("input"), item.get("output"))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    print(f"중복 제거 완료: {len(data)} -> {len(unique)} 항목")
    return unique

if __name__ == "__main__":
    final_dataset = generate_full_dataset(target_total=300, batch_size=30)
    
    # 중복 제거
    deduplicated_dataset = remove_duplicates(final_dataset)
    
    with open("eval_instruction_dataset_300.json", "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print("최종 평가 데이터셋 저장 완료!")