# Mixture of Experts (MoE) 기술 해설서

> 2026년 5월 기준 | 비전문가도 이해할 수 있는 완전 가이드

---

## 1부: MoE란 무엇인가

### 비유로 이해하기

**종합병원**을 떠올려 보세요.

환자(토큰)가 병원에 오면, 접수처(라우터)가 증상을 보고 적절한 전문의(Expert)에게 배정합니다. 내과, 외과, 안과, 피부과 등 수십 명의 전문의가 있지만, 한 환자가 모든 전문의를 만나지는 않습니다. 보통 1~2명의 전문의만 진료합니다.

이것이 MoE의 핵심입니다:
- **병원 전체 의사 수** = 전체 파라미터 수 (수조 개)
- **한 환자가 만나는 의사 수** = 활성 파라미터 수 (수백억 개)
- **접수처** = 라우터(Router/Gate)

반대로 **Dense 모델**은 "모든 환자가 모든 의사를 만나는 병원"입니다. 의사가 100명이면 환자 한 명당 100명 모두가 진료합니다. 정확할 수는 있지만, 엄청나게 비효율적이죠.

### 핵심 구성요소 3가지

```
입력 토큰 → [라우터] → Expert 2, Expert 7 선택 → 가중 합산 → 출력
              ↓
         (나머지 Expert들은 이 토큰에 대해 비활성)
```

**① Expert (전문가 네트워크)**
- 각각 독립된 Feed-Forward Network (FFN)
- 구조는 동일하지만, 학습 과정에서 서로 다른 패턴에 특화됨
- 모델 하나에 64~256개 이상의 Expert가 존재

**② Router (라우터/게이트)**
- 입력 토큰을 보고 "어떤 Expert가 이 토큰을 처리하기에 적합한가"를 점수로 매김
- 가장 높은 점수를 받은 상위 K개(보통 1~2개)의 Expert만 활성화
- 수식: `점수 = Softmax(토큰 × 라우터 가중치)` → 상위 K개 선택

**③ Sparse Activation (희소 활성화)**
- 전체 Expert 중 극소수만 활성화되는 것
- 예: 256개 Expert 중 2개만 활성화 = 전체의 0.8%만 연산에 참여
- 이것이 "파라미터 수는 크지만 연산량은 작다"를 가능하게 하는 핵심

### 왜 혁신인가: 파라미터 수 ≠ 연산량

| | Dense 모델 | MoE 모델 |
|---|---|---|
| 전체 파라미터 | 70B | 671B |
| 토큰당 활성 파라미터 | 70B (전부) | 37B (5.5%) |
| 토큰당 연산량 | 매우 큼 | Dense 70B와 비슷 |
| 모델 용량(지식) | 70B 수준 | 671B 수준 |

**결론**: MoE는 "작은 비용으로 큰 모델의 지식을 사용하는 방법"입니다.

### 흔한 오해 바로잡기

> ❌ "Expert 1은 수학 전문가, Expert 2는 코드 전문가처럼 깔끔하게 나뉜다"

실제로는 그렇지 않습니다. Expert의 전문화는 **통계적이고 창발적(emergent)**입니다. 특정 Expert가 "한국어 조사 처리"나 "숫자가 포함된 문장"에 강할 수는 있지만, 인간이 이해할 수 있는 깔끔한 카테고리로 나뉘지는 않습니다. 고차원 임베딩 공간에서의 영역 분할에 가깝습니다.

> ❌ "Expert가 많으면 추론이 느려진다"

아닙니다. 토큰당 활성화되는 Expert 수는 고정(보통 1~2개)이므로, Expert를 64개에서 256개로 늘려도 **토큰당 연산량은 동일**합니다. 다만 메모리에는 전체 Expert를 올려야 하므로 메모리 요구량은 증가합니다.


---

## 2부: MoE의 주요 문제들

MoE는 강력하지만, 구조적으로 6가지 핵심 문제를 안고 있습니다.

---

### 문제 1: 로드 불균형 & Expert Collapse (전문가 붕괴)

**비유**: 병원에 50명의 의사가 있는데, 접수처가 계속 같은 3명에게만 환자를 보냅니다. 나머지 47명은 놀고 있고, 3명은 과로합니다. 시간이 지나면 47명은 실력이 퇴화하고, 결국 3명짜리 병원이 됩니다.

**왜 발생하는가**:
- 학습 초기에 우연히 특정 Expert가 좋은 결과를 내면, 라우터는 그 Expert에 더 많은 토큰을 보냄
- 더 많은 토큰을 받은 Expert는 더 잘 학습됨 → 라우터가 더 선호 → **양의 피드백 루프**
- 결국 소수 Expert만 사용되고 나머지는 "죽은 Expert"가 됨

**무엇이 문제인가**:
- 전체 파라미터의 대부분이 낭비됨 (256개 Expert 중 10개만 실질적으로 사용)
- 모델 용량이 이론적 최대치에 한참 못 미침
- 과부하된 Expert는 다양한 입력을 처리해야 하므로 전문화가 약해짐

**왜 해결이 어려운가**:
- 강제로 균등 분배하면(auxiliary loss) Expert 전문화가 방해됨
- "균형"과 "전문화"는 본질적으로 상충하는 목표

---

### 문제 2: 학습 불안정성

**비유**: 접수처 직원(라우터)이 매일 배정 기준을 바꿉니다. 어제는 기침 환자를 내과로 보냈는데, 오늘은 외과로 보냅니다. 의사들은 혼란스럽고, 환자 치료 품질이 들쭉날쭉합니다.

**왜 발생하는가**:
- 라우터와 Expert가 동시에 학습되면서 서로 영향을 줌
- 라우터가 배정을 바꾸면 Expert가 받는 데이터 분포가 급변
- 특히 강화학습(RL) 단계에서 라우터의 불안정성이 증폭되어 학습 자체가 붕괴할 수 있음

**기존 해결책의 딜레마 (Auxiliary Loss)**:
- 로드 밸런싱을 위해 "모든 Expert를 균등하게 써라"는 보조 손실(auxiliary loss)을 추가
- 하지만 이 보조 손실이 너무 크면: Expert들이 과도하게 균일해져서 전문화 실패
- 너무 작으면: 로드 불균형 해결 안 됨
- 적절한 값을 찾기 어렵고, 모델/데이터마다 다름

---

### 문제 3: Expert 독립성의 한계

**비유**: 병원에서 내과 의사와 외과 의사가 한 환자를 동시에 보지만, 서로 대화하지 않습니다. 각자 소견서를 쓰고, 접수처가 두 소견서를 합칩니다. 복잡한 환자(예: 수술 후 내과 관리가 필요한 경우)에게는 이 방식이 부적절합니다.

**왜 발생하는가**:
- 기존 MoE에서 선택된 Expert들은 **병렬·독립적**으로 작동
- Expert A의 출력이 Expert B의 입력에 영향을 주지 않음
- 각 Expert는 동일한 원본 토큰만 보고 독립적으로 처리

**무엇이 문제인가**:
- 복잡한 추론(수학, 논리, 다단계 코딩)에서 Expert 간 협력이 불가능
- 모델의 "유효 깊이(effective depth)"가 제한됨
- Expert 조합의 다양성이 제한됨 (N개 중 K개 선택 = C(N,K) 조합만 가능)

---

### 문제 4: 추론 시 메모리 문제

**비유**: 병원에 의사 256명이 있습니다. 환자 한 명당 2명만 진료하지만, 256명 모두 출근해서 대기해야 합니다. 진료실(GPU 메모리)은 256명분이 필요하지만, 실제 진료(연산)는 2명분입니다.

**왜 발생하는가**:
- MoE 모델의 전체 파라미터가 메모리에 상주해야 함
- 어떤 토큰이 어떤 Expert를 호출할지 사전에 알 수 없으므로, 모든 Expert를 준비해야 함
- DeepSeek-V4-Pro: 1.6T 파라미터 전체를 메모리에 올려야 49B만 활성화

**무엇이 문제인가**:
- 671B MoE 모델은 FP16 기준 ~1.3TB 메모리 필요
- 단일 GPU(H100 80GB)로는 불가능, 최소 16~20개 GPU 필요
- 소비자 하드웨어에서는 실행 자체가 불가능
- Expert를 CPU로 오프로딩하면 CPU↔GPU 대역폭이 병목

---

### 문제 5: 파인튜닝 비효율

**비유**: 병원을 피부과 전문 클리닉으로 전환하려 합니다. 그런데 256명의 의사 전원에게 피부과 재교육을 시킵니다. 실제로 피부과 환자를 볼 의사는 20명뿐인데, 나머지 236명도 교육받느라 시간과 비용이 낭비됩니다.

**왜 발생하는가**:
- 기존 파인튜닝(LoRA 등)은 모든 Expert에 동일하게 adapter를 적용
- 하지만 특정 태스크에서 실제로 활성화되는 Expert는 전체의 일부
- "Cold Expert" (거의 호출되지 않는 Expert)에도 adapter를 붙이는 것은 낭비

**무엇이 문제인가**:
- 메모리 사용량이 Expert 수에 비례하여 증가
- 학습 시간이 불필요하게 길어짐
- Cold Expert의 adapter는 학습 데이터가 부족하여 오히려 노이즈를 유발할 수 있음

---

### 문제 6: 분산 학습의 복잡성

**비유**: 256명의 의사를 4개 건물(GPU)에 나눠 배치했습니다. 환자가 1번 건물에 도착했는데, 라우터가 3번 건물의 의사를 배정하면 환자를 3번 건물로 이송해야 합니다. 이송(통신) 시간이 진료(연산) 시간보다 길어질 수 있습니다.

**왜 발생하는가**:
- MoE의 Expert Parallelism은 Expert를 여러 GPU에 분산 배치
- 토큰이 특정 Expert에 라우팅되면, 해당 Expert가 있는 GPU로 토큰을 전송해야 함 (all-to-all 통신)
- 라우팅이 불균형하면 특정 GPU에 토큰이 몰려 tail latency 발생

**3중 제약의 결합**:
- **메모리**: 각 GPU에 올릴 수 있는 Expert 수 제한
- **통신**: GPU 간 토큰 이동 대역폭 제한
- **연산**: 불균형한 Expert 배치로 일부 GPU만 과부하

이 세 가지가 서로 얽혀 있어서, 하나를 최적화하면 다른 것이 악화되는 트레이드오프가 존재합니다.


---

## 3부: 각 문제별 최신 해결 기술 (2025~2026)

---

### 문제 1 해결: 로드 불균형 & Expert Collapse

---

#### Auxiliary-Loss-Free Load Balancing (DeepSeek-V3/V4)

**기존 방식의 문제부터 이해합시다.**

기존에는 "모든 Expert가 비슷한 수의 토큰을 받도록" 강제하는 보조 손실(auxiliary loss)을 사용했습니다. 이것은 메인 학습 목표(다음 토큰 예측)와는 별개의 추가 목표입니다. 문제는, 이 보조 손실이 메인 gradient에 간섭한다는 것입니다. 라우터 입장에서는 "이 토큰에 가장 적합한 Expert"로 보내고 싶은데, 보조 손실이 "아니, 저쪽 Expert가 한가하니까 저쪽으로 보내"라고 밀어붙입니다. 결과적으로 라우터가 최적이 아닌 배정을 하게 되고, 모델 성능이 떨어집니다.

**DeepSeek의 해결법은 놀랍도록 단순합니다.**

각 Expert에 하나의 숫자(bias term)를 추가합니다. 라우터가 토큰에 대해 Expert별 점수를 매길 때, 이 bias가 점수에 더해집니다. 핵심은 이 bias가 gradient로 학습되지 않는다는 것입니다. 대신, 단순한 규칙으로 업데이트됩니다:

- 특정 Expert가 평균보다 많은 토큰을 받았으면 → 그 Expert의 bias를 조금 낮춤
- 특정 Expert가 평균보다 적은 토큰을 받았으면 → 그 Expert의 bias를 조금 높임

이것이 전부입니다. 이 bias 조정은 메인 학습 gradient와 완전히 분리되어 있으므로, 라우터의 "이 토큰에 가장 적합한 Expert를 찾는" 본업을 전혀 방해하지 않습니다. 마치 온도 조절기처럼, 뜨거운 방(과부하 Expert)은 살짝 식히고 차가운 방(저활용 Expert)은 살짝 데우는 것입니다. 방 안에서 하는 일(Expert 학습)에는 전혀 간섭하지 않습니다.

DeepSeek-V3에서 이 방식을 처음 도입한 이후, V4와 후속 모델들에서 표준이 되었고, 이제 대부분의 새로운 MoE 모델이 이 접근법을 따릅니다.

---

#### Maximum Score Routing (2025)

**기존 방식의 문제를 먼저 봅시다.**

기존 MoE에서는 "capacity factor"라는 것을 설정합니다. 이것은 "각 Expert가 한 번에 처리할 수 있는 최대 토큰 수"입니다. 예를 들어 배치에 1000개 토큰이 있고 Expert가 10개면, 균등 분배 시 Expert당 100개입니다. Capacity factor를 1.2로 설정하면 Expert당 최대 120개까지 받을 수 있습니다.

문제는 두 가지입니다:
1. **Token dropping**: Expert가 120개를 초과하면 나머지 토큰은 그냥 버려집니다. 이 토큰들은 어떤 Expert도 처리하지 않으므로 정보가 손실됩니다.
2. **Padding 낭비**: 어떤 Expert는 30개만 받았는데 120개 크기의 텐서를 할당받습니다. 나머지 90개 자리는 빈 패딩으로 채워져 GPU 연산이 낭비됩니다.

**Maximum Score Routing은 이 capacity factor 자체를 없앱니다.**

대신 이렇게 작동합니다:
1. 모든 토큰에 대해 라우터가 점수를 매깁니다 (여기까지는 동일)
2. 토큰을 점수 순으로 정렬합니다
3. 각 Expert에 배정된 토큰들을 점수 순서대로 처리합니다
4. Expert당 처리량에 상한이 없으므로 토큰이 버려지지 않습니다
5. 빈 자리가 없으므로 패딩도 필요 없습니다

결과적으로 모든 토큰이 처리되고(정보 손실 없음), GPU 연산도 낭비 없이 사용됩니다. 단순하지만, 기존의 "고정 크기 배치" 가정을 깨는 발상의 전환입니다.

---

#### Routing-Free MoE (2026)

이것은 가장 급진적인 접근입니다. "라우터가 문제라면, 라우터를 아예 없애자."

기존 MoE에서 라우터는 학습이 어렵고, 불안정하고, 로드 불균형을 유발합니다. Routing-Free MoE는 라우터 대신 Expert 자체가 "이 토큰을 처리할지 말지"를 결정하게 합니다.

구체적으로, 각 Expert가 자체적인 활성화 함수를 가지고 있어서, 입력 토큰의 특성을 보고 스스로 "이건 내가 처리할 수 있다/없다"를 판단합니다. 중앙 집중식 배정(라우터)이 아니라 분산 자율 판단입니다.

추가로, "Expert 밸런싱"(각 Expert가 비슷한 양을 처리)과 "토큰 밸런싱"(각 토큰이 비슷한 수의 Expert에게 처리됨)을 하나의 통합 프레임워크에서 보간(interpolation)할 수 있게 설계되어, 상황에 따라 유연하게 조절 가능합니다.

아직 초기 연구 단계이지만, 라우터라는 구조적 병목을 근본적으로 제거한다는 점에서 주목받고 있습니다.

---

### 문제 2 해결: 학습 불안정성

---

#### SimSMoE — Representational Collapse 해결 (NAACL 2025)

**"Representational Collapse"가 뭔지부터 설명합니다.**

MoE에서 여러 Expert가 학습을 진행하다 보면, 서로 다른 Expert인데도 거의 동일한 출력을 내는 현상이 발생합니다. Expert 1에 토큰을 넣든 Expert 5에 넣든 결과가 비슷해지는 것입니다. 이렇게 되면 256개 Expert가 있어도 실질적으로는 10개짜리 모델과 다를 바 없습니다. 이것이 representational collapse입니다.

**왜 발생하는가?** 학습 초기에 Expert들이 비슷한 초기값에서 시작하고, 비슷한 데이터를 받으면, gradient도 비슷해져서 비슷한 방향으로 업데이트됩니다. 시간이 지나도 이 유사성에서 벗어나지 못하는 것입니다.

**SimSMoE의 해결법:**

학습 중에 주기적으로 Expert 쌍의 출력 유사도를 측정합니다. 구체적으로, 같은 토큰을 Expert A와 Expert B에 넣었을 때 출력 벡터의 코사인 유사도를 계산합니다. 이 유사도가 높으면(즉, 두 Expert가 비슷한 출력을 내면) 페널티를 부여합니다.

이 페널티는 Expert들을 "서로 밀어내는" 효과를 줍니다. Expert A가 특정 방향으로 학습하고 있으면, Expert B는 그 방향을 피해서 다른 방향으로 학습하게 됩니다. 결과적으로 각 Expert가 서로 다른 표현 공간의 영역을 담당하게 되어, 전체 모델의 표현력이 극대화됩니다.

기존 auxiliary loss와의 차이점: auxiliary loss는 "토큰 배정 수"를 균등하게 만들려 하지만, SimSMoE는 "Expert의 출력 표현"을 다양하게 만들려 합니다. 목표 자체가 다릅니다.

---

#### MoE에서의 강화학습 안정화 (2025)

**왜 MoE + RL이 특히 불안정한가?**

대형 언어모델은 사전학습 후 RLHF(인간 피드백 강화학습)를 거칩니다. Dense 모델에서는 이 과정이 비교적 안정적입니다. 하지만 MoE에서는 심각한 문제가 생깁니다.

RL 학습에서는 "이전 정책(policy)으로 생성한 데이터"를 "현재 정책"으로 평가합니다. Dense 모델에서는 정책이 바뀌어도 같은 파라미터가 활성화되므로 변화가 점진적입니다. 하지만 MoE에서는 정책이 조금만 바뀌어도 라우터의 배정이 달라질 수 있습니다. 어제는 토큰 X가 Expert 3으로 갔는데, 오늘은 Expert 7로 갑니다. 이렇게 되면 "이전 정책으로 생성한 데이터"와 "현재 정책"의 괴리가 급격히 커져서, gradient의 분산이 폭발하고 학습이 발산합니다.

**해결법: Router-Aware Importance Sampling**

Off-policy RL에서는 importance sampling(IS) 가중치를 사용하여 이전 정책과 현재 정책의 차이를 보정합니다. 이 논문은 IS 가중치를 계산할 때 라우터의 logit(배정 점수)을 함께 고려합니다.

구체적으로:
1. 이전 시점에서 토큰이 어떤 Expert로 라우팅되었는지 기록합니다
2. 현재 시점에서 같은 토큰의 라우팅 확률을 계산합니다
3. 라우팅이 크게 바뀐 토큰(예: Expert 3 → Expert 7)의 IS 가중치를 축소합니다
4. 라우팅이 유지된 토큰의 IS 가중치는 정상 적용합니다

이렇게 하면 "라우팅 변경으로 인한 급격한 gradient"가 억제되어, 학습이 안정적으로 진행됩니다. MoE 모델의 RLHF 단계에서 발생하던 catastrophic collapse(학습 중 갑자기 모델이 무의미한 출력을 내기 시작하는 현상)를 방지합니다.


---

### 문제 3 해결: Expert 독립성의 한계

---

#### Chain-of-Experts (CoE) — 2026, Northwestern/Oxford

**기존 MoE의 작동 방식을 다시 봅시다.**

토큰 X가 들어오면:
1. 라우터가 Expert 2와 Expert 7을 선택합니다
2. Expert 2가 X를 처리하여 출력 A를 냅니다
3. Expert 7이 **같은 X**를 처리하여 출력 B를 냅니다 (A를 모릅니다)
4. 최종 출력 = 0.6×A + 0.4×B (가중 합산)

핵심: Expert 2와 Expert 7은 서로의 존재를 모릅니다. 둘 다 원본 X만 봅니다.

**CoE는 이것을 순차적으로 바꿉니다.**

같은 2개 Expert를 사용하지만, 순서대로 처리합니다:
1. 라우터 ①이 Expert 2를 선택합니다
2. Expert 2가 X를 처리하여 중간 결과 X'를 냅니다
3. 라우터 ②가 **X'를 보고** Expert 7을 선택합니다 (원본 X가 아니라 중간 결과를 봅니다)
4. Expert 7이 **X'를 처리**하여 최종 출력을 냅니다

차이가 보이시나요? Expert 7은 "Expert 2가 이미 처리한 결과"를 받아서 작업합니다. Expert 2의 작업 위에 Expert 7이 추가 작업을 하는 것입니다. 마치 편집자가 초고를 쓰고, 교정자가 그 초고를 다듬는 것과 같습니다.

**왜 이것이 더 강력한가?**

1. **정보 흐름**: Expert 7이 Expert 2의 출력을 보므로, 두 Expert가 상호보완적으로 작동할 수 있습니다. Expert 2가 "대략적인 방향"을 잡으면, Expert 7이 "세부 조정"을 합니다.

2. **적응적 라우팅**: 두 번째 라우터는 중간 결과 X'를 보고 Expert를 선택합니다. 즉, "첫 번째 Expert가 무엇을 했는지"에 따라 두 번째 Expert 선택이 달라집니다. 기존 MoE에서는 불가능한 동적 조합입니다.

3. **조합 폭발**: 기존 MoE에서 64개 중 8개를 고르면 C(64,8) = 약 44억 가지 조합입니다. CoE에서 2번의 iteration으로 4개씩 고르면 C(64,4)² = 약 40조 가지 조합입니다. 823배 더 다양한 Expert 조합이 가능합니다.

**실험 결과:**
- 수학 추론 validation loss: 1.20 → 1.12 (동일 연산량에서)
- 12개 레이어 MoE와 동일한 성능을 4개 레이어 CoE로 달성 (메모리 42% 절감)
- 16개 Expert 선택 MoE보다 8개 Expert 2-iteration CoE가 더 좋은 성능

**한계**: 순차 처리이므로 병렬성이 줄어들어 실제 wall-clock 시간은 약간 증가합니다. 이론적 FLOP은 동일하지만, GPU의 병렬 행렬곱 효율이 떨어집니다.

---

#### Super Experts 발견 (ICLR 2026)

이것은 "해결 기술"이라기보다 "중요한 발견"입니다. 하지만 모델 압축과 배포 전략을 근본적으로 바꾸는 발견이라 포함합니다.

**발견 내용:**

연구자들이 MoE 모델의 Expert들을 분석하다가 이상한 패턴을 발견했습니다. 수천 개의 Expert 중 극소수(3~10개)가 다른 Expert들과 질적으로 다른 행동을 보였습니다. 이 Expert들은 down_proj 레이어에서 극단적으로 큰 활성화 값(outlier)을 생성합니다. 다른 Expert들의 출력이 -1~+1 범위라면, 이 Expert들은 -100~+100 수준의 값을 냅니다.

**왜 중요한가?**

연구자들이 Qwen3-30B-A3B 모델(6,144개 Expert)에서 이 Super Expert 3개만 제거했더니, 모델이 완전히 붕괴했습니다. 반복적이고 무의미한 텍스트만 생성하게 되었고, 수학과 추론 벤치마크 점수가 급락했습니다. 반면, 일반 Expert 수십 개를 제거해도 성능 저하는 미미했습니다.

**이것이 의미하는 바:**

1. **프루닝(가지치기)**: MoE 모델을 작게 만들려고 Expert를 제거할 때, Super Expert를 건드리면 안 됩니다. "어떤 Expert가 Super Expert인가"를 먼저 식별해야 합니다.

2. **양자화**: Super Expert는 극단적 활성화 값을 생성하므로, 일반적인 양자화(FP16→INT4)를 적용하면 이 값들이 잘려나가 모델이 망가집니다. Super Expert는 더 높은 정밀도로 유지해야 합니다.

3. **모델 이해**: Super Expert는 레이어 간 정보 흐름을 안정화하는 "앵커" 역할을 하는 것으로 보입니다. Attention sink(특정 토큰에 attention이 집중되는 현상)과 유사한 메커니즘입니다.

---

### 문제 4 해결: 추론 시 메모리 문제

---

#### Predictive Expert Caching + Token Scheduling (2025)

**문제를 구체적 숫자로 봅시다.**

DeepSeek-V3 (671B 파라미터)를 FP16으로 로드하면 약 1.3TB 메모리가 필요합니다. H100 GPU 한 장은 80GB입니다. 16장을 써도 1.28TB로 빠듯합니다. 하지만 실제로 한 토큰이 활성화하는 Expert는 전체의 5% 미만입니다. 나머지 95%는 "혹시 호출될까 봐" 메모리에 대기하고 있을 뿐입니다.

**Expert Offloading의 기본 아이디어:**

자주 쓰이지 않는 Expert를 GPU 메모리에서 CPU 메모리(DRAM)로 내립니다. 호출될 때만 CPU→GPU로 올립니다. 이렇게 하면 GPU에는 자주 쓰이는 Expert만 상주하므로 메모리를 크게 절약할 수 있습니다.

**문제**: CPU→GPU 전송 속도가 병목입니다. PCIe 4.0 기준 약 32GB/s인데, Expert 하나가 수백 MB라면 전송에 수~수십 ms가 걸립니다. 토큰 생성은 ms 단위여야 하므로, 매번 전송하면 추론이 극도로 느려집니다.

**Predictive Expert Caching의 해결법:**

핵심 관찰: MoE에서 레이어 N의 라우팅 패턴은 레이어 N+1의 라우팅 패턴과 상관관계가 있습니다. 즉, 현재 레이어에서 어떤 Expert가 활성화되었는지를 보면, 다음 레이어에서 어떤 Expert가 활성화될지를 높은 확률로 예측할 수 있습니다.

작동 방식:
1. 레이어 N을 처리하는 동안, 예측 모델이 "레이어 N+1에서 활성화될 Expert"를 예측합니다
2. 레이어 N의 연산과 **동시에(비동기적으로)** 예측된 Expert를 CPU→GPU로 전송합니다
3. 레이어 N+1 처리가 시작될 때, 필요한 Expert가 이미 GPU에 있습니다
4. 예측이 틀린 경우에만 동기적 전송이 발생합니다 (캐시 미스)

추가로, Token Scheduling이 캐시 히트율을 높입니다. 여러 토큰을 처리할 때, 같은 Expert를 사용하는 토큰들을 묶어서 연속 처리합니다. 이렇게 하면 Expert를 한 번 GPU에 올려놓고 여러 토큰을 처리할 수 있어 전송 횟수가 줄어듭니다.

**성과**: 단일 GPU에서 GPU 메모리 사용량 93.7% 절감, 추론 처리량 10배 향상. 소비자 GPU(RTX 4090 24GB)에서도 수백 B 파라미터 MoE 모델 실행이 가능해집니다.

---

#### Fine-Grained Expert Offloading (2025)

기존 오프로딩은 "Expert 전체"를 단위로 CPU↔GPU를 오갑니다. Expert 하나가 수백 MB이므로 전송 단위가 큽니다.

이 연구는 Expert를 더 작은 단위로 쪼갭니다. Expert 내부의 파라미터를 "자주 쓰이는 부분"과 "가끔 쓰이는 부분"으로 나누어, 자주 쓰이는 부분만 GPU에 상주시킵니다.

구체적으로, Expert의 FFN은 보통 up_proj → activation → down_proj 구조입니다. 분석 결과, 특정 뉴런(열/행)이 대부분의 토큰에서 활성화되고, 나머지는 드물게 활성화됩니다. 자주 활성화되는 뉴런만 GPU에 두고, 나머지는 CPU에 둡니다.

이렇게 하면 "Expert 전체를 올리거나 내리거나"의 이분법에서 벗어나, 더 세밀한 메모리-지연시간 트레이드오프가 가능합니다.


---

### 문제 5 해결: 파인튜닝 비효율

---

#### MoE-Sieve — Routing-Guided LoRA (2026)

**LoRA가 뭔지 간단히 복습합시다.**

LoRA(Low-Rank Adaptation)는 거대 모델을 파인튜닝할 때, 원본 가중치는 동결하고 작은 "adapter 행렬"만 학습하는 기법입니다. 원본 모델이 100GB라면, LoRA adapter는 수백 MB 수준이므로 메모리와 시간을 크게 절약합니다.

**MoE에 LoRA를 적용할 때의 문제:**

MoE 모델에 LoRA를 적용하면, 보통 모든 Expert에 adapter를 붙입니다. Expert가 256개면 adapter도 256개입니다. 하지만 실제로 특정 태스크(예: 한국어 의료 QA)를 수행할 때, 256개 Expert 중 실제로 활성화되는 것은 일부입니다. 나머지 Expert의 adapter는 학습 데이터를 거의 받지 못하므로 제대로 학습되지 않고, 오히려 노이즈가 될 수 있습니다.

**MoE-Sieve의 접근법:**

3단계로 작동합니다:

**Step 1 — Profile (프로파일링):**
파인튜닝할 데이터셋의 일부(수백~수천 샘플)를 모델에 통과시키면서, 각 레이어에서 어떤 Expert가 얼마나 자주 활성화되는지를 기록합니다. 이 과정은 forward pass만 하면 되므로 매우 빠릅니다 (전체 파인튜닝 시간의 1% 미만).

결과 예시 (레이어 15):
- Expert 3: 토큰의 23% 처리 (Hot)
- Expert 7: 토큰의 18% 처리 (Hot)
- Expert 12: 토큰의 15% 처리 (Hot)
- Expert 1: 토큰의 0.3% 처리 (Cold)
- Expert 45: 토큰의 0.1% 처리 (Cold)
- ...

**Step 2 — Select (선택):**
각 레이어에서 상위 25%의 "Hot Expert"만 선택합니다. 나머지 75%의 "Cold Expert"는 파인튜닝 대상에서 제외합니다.

**Step 3 — Fine-tune (파인튜닝):**
선택된 Hot Expert에만 LoRA adapter를 붙이고 학습합니다. Cold Expert는 원본 가중치 그대로 동결됩니다.

**왜 25%만으로 충분한가?**

연구 결과, MoE의 라우팅은 태스크별로 매우 편향되어 있습니다. 특정 태스크에서는 소수의 Expert가 대부분의 토큰을 처리합니다. 이 소수의 Expert만 태스크에 맞게 조정하면, 나머지는 건드리지 않아도 됩니다. 오히려 Cold Expert에 adapter를 붙이면, 학습 데이터가 부족하여 adapter가 제대로 수렴하지 못하고 노이즈를 추가하는 역효과가 있었습니다.

**성과**: 전체 Expert에 LoRA를 적용한 것과 동등한 성능을 25%의 adapter만으로 달성. 메모리 75% 절감, 학습 시간 단축.

---

#### L-MoE — Expert 자체를 LoRA로 재정의 (2025)

이것은 더 근본적인 접근입니다. "기존 Expert에 LoRA를 붙이는" 것이 아니라, "Expert 자체를 LoRA adapter로 만드는" 것입니다.

기존 MoE에서 Expert는 큰 FFN(수억 파라미터)입니다. L-MoE에서 Expert는 작은 LoRA adapter(수백만 파라미터)입니다. 게이팅 네트워크가 여러 LoRA adapter의 가중 평균을 계산하여, 토큰별로 다른 "가상의 Expert"를 동적으로 합성합니다.

장점:
- Expert당 파라미터가 극도로 작으므로 Expert 수를 크게 늘릴 수 있음
- 가중 평균으로 조합하므로, N개의 LoRA adapter로 사실상 무한한 "가상 Expert"를 만들 수 있음
- End-to-end 학습이 가능하여 게이팅과 adapter가 동시에 최적화됨

예를 들어, 기존 MoE가 64개의 큰 Expert(각 1억 파라미터)를 가진다면, L-MoE는 256개의 작은 LoRA adapter(각 100만 파라미터)를 가질 수 있습니다. 총 파라미터는 2.56억으로 기존의 64억보다 훨씬 작지만, 조합의 다양성은 훨씬 큽니다. 토큰 A에는 adapter 3, 7, 12의 가중 평균이 적용되고, 토큰 B에는 adapter 1, 5, 45의 가중 평균이 적용되는 식입니다.

이 접근법은 특히 "기존 Dense 모델을 MoE로 변환"할 때 유용합니다. 사전학습된 Dense 모델의 FFN을 동결하고, 그 위에 L-MoE adapter들을 추가하면 파인튜닝만으로 MoE의 이점을 얻을 수 있습니다.

---

### 문제 6 해결: 분산 학습의 복잡성

---

#### Megatron Core MoE (2026, NVIDIA)

**MoE 분산 학습이 왜 Dense 모델보다 어려운지 구체적으로 봅시다.**

Dense 모델의 분산 학습은 비교적 예측 가능합니다. 모델을 레이어 단위로 자르거나(Pipeline Parallelism), 레이어 내부를 나누거나(Tensor Parallelism), 데이터를 나누면(Data Parallelism) 됩니다. 각 GPU의 부하가 균등하고, 통신 패턴도 규칙적입니다.

MoE에서는 **Expert Parallelism**이라는 새로운 차원이 추가됩니다. Expert를 여러 GPU에 나눠 배치하는 것입니다. 문제는 이것이 "all-to-all 통신"을 유발한다는 점입니다.

**All-to-all 통신이란:**

8개 GPU에 Expert가 분산되어 있다고 합시다. GPU 1에 있는 토큰이 GPU 5의 Expert로 라우팅되면, 이 토큰을 GPU 1→GPU 5로 보내야 합니다. 동시에 GPU 3의 토큰은 GPU 1로, GPU 7의 토큰은 GPU 2로... 모든 GPU가 모든 GPU에게 데이터를 보내는 패턴입니다. 이것은 통신량이 GPU 수의 제곱에 비례하여 증가하므로, 스케일링이 매우 어렵습니다.

게다가 라우팅이 불균형하면, 특정 GPU에 토큰이 몰립니다. GPU 5에 인기 Expert가 있으면 모든 GPU가 GPU 5로 토큰을 보내고, GPU 5는 과부하되어 전체 학습의 병목이 됩니다.

**Megatron Core MoE의 접근법:**

NVIDIA는 이 문제를 "결합 제약(coupled constraints)"으로 모델링합니다. 메모리, 통신, 연산이 서로 얽혀 있으므로, 하나만 최적화하면 안 되고 세 가지를 동시에 고려해야 합니다.

구체적으로:
1. **Expert 배치 최적화**: 어떤 Expert를 어떤 GPU에 놓을지를 통신 패턴 분석을 통해 결정합니다. 자주 함께 호출되는 Expert들을 같은 GPU에 배치하면 inter-GPU 통신이 줄어듭니다.

2. **통신-연산 오버랩**: all-to-all 통신을 하는 동안 다른 연산(attention 등)을 동시에 수행합니다. 통신이 끝나기를 기다리지 않고, 통신과 연산을 파이프라인으로 겹칩니다.

3. **동적 부하 분산**: 학습 중 라우팅 패턴이 변하므로, 주기적으로 Expert 배치를 재조정합니다.

이 시스템은 DeepSeek-V4급 모델(1.6T 파라미터)의 학습을 수천 개 GPU에서 효율적으로 수행할 수 있게 합니다.

---

#### FSMoE — 유연하고 확장 가능한 MoE 학습 시스템 (2025)

**문제: MoE 구현마다 최적화 방법이 다르다**

DeepSeek의 MoE와 Mixtral의 MoE는 구조가 다릅니다. Expert 수, 라우팅 방식, shared expert 유무, 통신 패턴이 모두 다릅니다. 기존에는 각 MoE 구현마다 별도의 최적화를 수작업으로 해야 했습니다.

**FSMoE의 접근법:**

MoE 레이어의 작동을 4개의 독립 모듈로 추상화합니다:
1. **Token Routing**: 토큰을 Expert에 배정하는 단계
2. **Token Communication**: 토큰을 해당 Expert가 있는 GPU로 전송하는 단계
3. **Expert Computation**: Expert가 토큰을 실제로 처리하는 단계
4. **Expert Parallelism**: Expert를 GPU 간에 분산하는 방식

각 모듈의 실행 시간을 온라인으로 프로파일링(실시간 측정)합니다. 그리고 이 측정값을 바탕으로 태스크 스케줄링을 최적화합니다. 예를 들어:
- Token Communication이 병목이면 → 통신과 연산의 오버랩을 늘림
- Expert Computation이 병목이면 → Expert를 더 많은 GPU에 분산
- Token Routing이 병목이면 → 라우팅 연산을 비동기로 처리

이 추상화 덕분에, 새로운 MoE 아키텍처가 나와도 4개 모듈의 구현만 바꾸면 최적화 프레임워크를 그대로 재사용할 수 있습니다.

---

#### MoETuner — Expert-to-GPU 최적 배치 (2025)

**구체적인 문제 상황:**

32개 GPU에 256개 Expert를 배치한다고 합시다. 단순하게 GPU당 8개씩 균등 배치하면 될까요? 아닙니다. 라우팅 패턴에 따라 결과가 크게 달라집니다.

예를 들어, Expert 1과 Expert 2가 항상 같은 토큰에 의해 함께 호출된다면, 이 둘을 같은 GPU에 놓는 것이 좋습니다. 토큰이 GPU 간 이동할 필요가 없으니까요. 반대로, Expert 1과 Expert 2가 절대 동시에 호출되지 않는다면, 같은 GPU에 놓으면 한쪽이 항상 놀게 됩니다.

**MoETuner의 접근법:**

이것을 최적화 문제로 정식화합니다:
- **목적 함수 1**: Inter-GPU 토큰 라우팅 비용 최소화 (함께 호출되는 Expert를 같은 GPU에)
- **목적 함수 2**: GPU 간 토큰 처리량 균형 (특정 GPU에 인기 Expert가 몰리지 않게)
- **제약 조건**: 각 GPU의 메모리 용량

이 두 목적은 상충합니다. 인기 Expert들을 한 GPU에 모으면 통신은 줄지만 부하가 불균형해집니다. MoETuner는 이 트레이드오프의 최적점을 찾아 Expert를 배치합니다.

결과적으로 tail latency(가장 느린 GPU가 끝나기를 기다리는 시간)가 크게 줄어들고, 전체 학습 시간이 단축됩니다.

---

### 3부 요약: 문제 → 해결의 핵심 아이디어

| 문제 | 핵심 해결 아이디어 | 대표 기술 |
|------|-----------------|----------|
| 로드 불균형 | 메인 학습을 방해하지 않는 별도 메커니즘으로 균형 조절 | Auxiliary-Loss-Free (bias 조정) |
| 학습 불안정 | Expert 간 표현 다양성 강제 + 라우팅 변화에 대한 gradient 보정 | SimSMoE, Router-Aware IS |
| Expert 독립성 | 병렬 처리를 순차 처리로 전환하여 Expert 간 정보 전달 | Chain-of-Experts |
| 메모리 문제 | 다음에 필요한 Expert를 예측하여 미리 로드 | Predictive Expert Caching |
| 파인튜닝 낭비 | 실제 활성화되는 Expert만 식별하여 선택적 적응 | MoE-Sieve |
| 분산 학습 | 통신·연산·메모리를 동시에 고려하는 통합 최적화 | Megatron Core MoE |


---

## 4부: 산업계 적용 현황 (2026년 5월 기준)

### 현재 상황: "MoE가 기본값이 된 시대"

2026년 4월 기준, 프론티어 AI 모델의 대다수가 MoE 아키텍처를 채택했습니다. Dense 모델로 남아있는 주요 프론티어 모델은 Anthropic의 Claude 시리즈 정도입니다.

### 주요 모델 비교

| 모델 | 총 파라미터 | 활성 파라미터 | 활성 비율 | Expert 구성 | 핵심 혁신 |
|------|-----------|------------|----------|------------|----------|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 3% | 256 routed + shared | Auxiliary-loss-free 밸런싱, Hybrid Attention (CSA+HCA), 1M 컨텍스트 |
| **DeepSeek-V4-Flash** | 284B | 13B | 4.6% | Fine-grained MoE | 경량 고속 추론 특화 |
| **Llama 4 Maverick** | ~400B | 17B | 4.3% | 128 routed + 1 shared | 네이티브 멀티모달, interleaved dense/MoE 레이어 |
| **Llama 4 Scout** | 109B | 17B | 15.6% | MoE | 10M 토큰 컨텍스트, 단일 H100 실행 가능 |
| **Qwen3-235B** | 235B | 22B | 9.4% | 128 expert pool, top-8 | 강력한 다국어, 에이전트 워크플로우 |
| **Mistral Large 3** | 675B | 41B | 6% | 256K 컨텍스트 | 단일 8-GPU 노드 배포, 코딩 최강 |
| **Kimi K2** | ~1T | ~32B | 3.2% | 네이티브 멀티모달 | 에이전트 특화 |
| **GPT-5/GPT-OSS** | 비공개 | 비공개 | - | 멀티모달 MoE | 코딩/수학/대화/비전 Expert 분리 |

### 설계 철학의 분화

**DeepSeek 계열 — "극단적 세분화"**
- 256개의 작은 Expert + auxiliary-loss-free 밸런싱
- Expert를 매우 작게 만들어 세밀한 전문화 유도
- 비용 효율 극대화 (V3 학습 비용 $5.5M으로 화제)

**Meta Llama 4 — "하이브리드 구조"**
- Dense 레이어와 MoE 레이어를 교차 배치 (interleaved)
- Shared Expert로 전역 지식 보존 + Routed Expert로 전문화
- 멀티모달을 아키텍처 수준에서 네이티브 통합

**Qwen3 — "균형잡힌 확장"**
- 128개 Expert에서 8개 활성화 (중간 수준의 sparsity)
- 다국어 + 에이전트 + 도구 사용에 균형 있는 설계
- 오픈소스 생태계에서 가장 범용적

**Mistral — "배포 효율성"**
- 단일 8-GPU 노드에서 실행 가능한 설계 목표
- FP8 양자화 + 최적화된 서빙 스택
- 실용적 배포를 최우선으로 설계

### 2026년 MoE가 향하는 방향

1. **Routing-Free 설계**: 라우터 자체를 제거하는 근본적 재설계
2. **Super Expert 보존 압축**: 핵심 Expert를 식별하고 보존하면서 나머지를 공격적으로 압축
3. **네이티브 멀티모달**: 텍스트/이미지/비디오/코드를 모달리티별 Expert 경로로 처리
4. **MoE 얽힘(Entanglement) 활용**: 레이어 간 라우팅 의존성을 활용한 전체 경로 최적화
5. **EMO (Emergent Modularity)**: 사전 정의 없이 데이터에서 모듈 구조가 자연 발현되는 학습


---

## 5부: AWS에서 MoE 활용하기

### 목적별 서비스 매핑

```
┌─────────────────────────────────────────────────────────────────┐
│                    "나는 MoE 모델을..."                           │
├─────────────────┬───────────────────┬───────────────────────────┤
│  쓰기만 하면 된다  │  파인튜닝하고 싶다  │  처음부터 학습하고 싶다     │
│                 │                   │                           │
│  Amazon Bedrock │  SageMaker AI     │  SageMaker + Trn2/P5     │
│  (API 호출)     │  + Multi-LoRA     │  + Expert Parallelism    │
└─────────────────┴───────────────────┴───────────────────────────┘
```

---

### 계층 1: MoE 모델 사용 — Amazon Bedrock

**"인프라 걱정 없이 API로 MoE 모델을 호출하고 싶다"**

Bedrock은 MoE의 가장 큰 배포 난제(거대한 메모리 요구량)를 완전히 추상화합니다. 사용자는 활성 파라미터 기준의 토큰 비용만 지불합니다.

| Bedrock에서 사용 가능한 MoE 모델 | Bedrock 출시일 | 용도 | 링크 |
|---|---|---|---|
| **DeepSeek-R1** | 2025.03.10 (서버리스 GA) | 수학/코딩/추론 | [블로그](https://aws.amazon.com/blogs/aws/deepseek-r1-now-available-as-a-fully-managed-serverless-model-in-amazon-bedrock/) |
| **Llama 4 Maverick / Scout** | 2025.04.28 (서버리스 GA) | 멀티모달 이해, 장문맥 분석 | [블로그](https://aws.amazon.com/blogs/aws/llama-4-models-from-meta-now-available-in-amazon-bedrock-serverless/) |
| **Mistral Large 3** | 2025.12 (Bedrock open weight 확장) | 코딩, 256K 컨텍스트 작업 | [발표](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-fully-managed-open-weight-models/) |
| **Mixtral 8x7B** | 2024 (초기 지원) | 비용 효율적 범용 작업 | [문서](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html) |
| **Pixtral Large** | 2025.12 | 비전+텍스트 멀티모달 | [모델 카드](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-pixtral-large.html) |

**핵심 가치**: 671B 파라미터 모델을 GPU 한 장 없이 사용 가능. MoE의 "활성 파라미터만 비용 지불" 특성이 서버리스 과금과 자연스럽게 맞물림.

---

### 계층 2: MoE 모델 파인튜닝 — SageMaker AI

**"오픈소스 MoE 모델을 내 데이터에 맞게 커스터마이징하고 싶다"**

#### Multi-LoRA on MoE (vLLM 기반) — 2026.02.26 출시
- 하나의 MoE 모델에 여러 LoRA adapter를 동시 서빙
- 5개 고객의 커스텀 모델을 단일 GPU에서 처리
- Expert별 선택적 adapter 적용 가능 (MoE-Sieve 방식과 호환)
- [AWS 블로그](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/) | [vLLM 블로그](https://vllm.ai/blog/multi-lora)

#### Custom Model Import
- DeepSeek-R1, Llama 4 등 오픈소스 MoE 모델을 직접 임포트
- Bedrock의 보안/가드레일/모니터링 기능 그대로 활용
- 자체 파인튜닝한 모델을 프로덕션 서빙 인프라에 즉시 배포
- [DeepSeek-R1 임포트 가이드](https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-llama-models-in-amazon-bedrock/)

---

### 계층 3: MoE 모델 학습 — SageMaker + 분산 학습

**"MoE 모델을 처음부터 사전학습하거나 대규모 커스터마이징을 하고 싶다"**

#### SageMaker Model Parallelism Library v2 (SMP v2) — 2023.12 출시, Expert Parallelism 2024.05 추가

MoE 학습에 필요한 Expert Parallelism을 네이티브 지원합니다:

| 병렬화 기법 | 역할 | MoE에서의 의미 |
|---|---|---|
| **Expert Parallelism** | Expert를 GPU 간 분산 | 256개 Expert를 32개 GPU에 8개씩 배치 |
| **Tensor Parallelism** | 단일 레이어를 GPU 간 분할 | 큰 Expert를 여러 GPU에 걸쳐 배치 |
| **Pipeline Parallelism** | 레이어를 순차적으로 분할 | 깊은 MoE 모델의 메모리 분산 |
| **Data Parallelism** | 데이터를 GPU 간 분할 | 학습 처리량 증가 |

이 4가지를 조합하여 "4D Parallelism"으로 조 단위 파라미터 MoE 모델 학습이 가능합니다.

- [Expert Parallelism 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-expert-parallelism.html)
- [Mixtral 8x7B 사전학습 가이드 (2024.05)](https://aws.amazon.com/blogs/machine-learning/accelerate-mixtral-8x7b-pre-training-with-expert-parallelism-on-amazon-sagemaker/)
- [SMP v2 소개 블로그 (2024.04)](https://aws.amazon.com/blogs/machine-learning/distributed-training-and-efficient-scaling-with-the-amazon-sagemaker-model-parallel-and-data-parallel-libraries/)

#### Flexible Training Plans
- GPU 용량을 시간 단위로 예약
- MoE 학습처럼 수일~수주 걸리는 작업에 안정적 자원 확보
- 학습 중단 없이 연속 실행 보장

---

### 계층 4: 하드웨어 인프라

#### AWS Trainium — MoE 최적화 칩

| 칩 세대 | MoE 관련 특징 | 출시일 | 링크 |
|---|---|---|---|
| **Trainium3 (Trn3 UltraServer)** | MoE를 1순위 워크로드로 명시. Trn2 대비 4.4배 성능, 3.9배 메모리 대역폭. 3nm 공정. | 2025.12 GA (re:Invent 2025) | [발표](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/) / [제품 페이지](https://aws.amazon.com/ec2/instance-types/trn3/) |
| **Trainium2 (Trn2)** | 조 단위 파라미터 모델 학습 지원 | 2024.12 GA (re:Invent 2024) | [제품 페이지](https://aws.amazon.com/ec2/instance-types/trn2/) |
| **Inferentia2 (Inf2)** | 추론 특화. 4배 처리량, 10배 낮은 지연시간 | 2023 GA | [제품 페이지](https://aws.amazon.com/ai/machine-learning/inferentia/) |

**Trn3가 MoE에 중요한 이유**:
- MoE의 all-to-all 통신 패턴에 최적화된 인터커넥트
- Expert 오프로딩 없이 대규모 MoE를 온칩에서 처리할 수 있는 메모리 대역폭
- 공식 발표에서 "Mixture-of-Experts, reinforcement learning, reasoning, long-context"를 핵심 타겟으로 명시

#### Neuron SDK — MoE 소프트웨어 스택

| 버전 | MoE 기능 | 출시일 | 링크 |
|---|---|---|---|
| **Neuron SDK 2.26** | Expert Parallelism 지원 (beta), NKI API | 2025.09 | [발표](https://aws.amazon.com/about-aws/whats-new/2025/09/aws-neuron-2-26-announce) |
| **Neuron SDK 2.29** | NKI Standard Library 공개, 저수준 커널 최적화 가능 | 2026.04 | [발표](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/) |

NKI(Neuron Kernel Interface)를 통해 MoE 라우팅 커널을 직접 최적화할 수 있어, 커스텀 라우팅 전략 구현이 가능합니다.

#### GPU 인스턴스 + 네트워킹

| 서비스 | MoE에서의 역할 | 링크 |
|---|---|---|
| **EC2 P5 (H100) / P5e (H200)** | MoE 모델 학습 및 서빙용 최고 성능 GPU | [P5](https://aws.amazon.com/ec2/instance-types/p5/) |
| **Elastic Fabric Adapter (EFA)** | Expert Parallelism의 all-to-all 통신 가속 (400Gbps) | [EFA](https://aws.amazon.com/hpc/efa/) |
| **FSx for Lustre** | 수 TB MoE 체크포인트의 고속 저장/로드 | [FSx](https://aws.amazon.com/fsx/lustre/) |
| **Deep Learning Containers** | vLLM + MoE 모델 배포용 사전 구성 컨테이너 (EKS/ECS) | [DLC 가이드](https://aws.github.io/deep-learning-containers/tutorials/vllm-samples/deepseek/eks/) |

---

### 정리: 문제 ↔ AWS 서비스 매핑

| MoE 문제 (2부) | AWS 해결 방안 |
|---|---|
| 추론 시 메모리 문제 | Bedrock (서버리스 추상화), Inf2 (추론 최적화 칩) |
| 파인튜닝 비효율 | SageMaker Multi-LoRA (선택적 Expert adapter) |
| 분산 학습 복잡성 | SageMaker SMP v2 (Expert Parallelism), EFA (통신 가속) |
| 로드 불균형 | Neuron SDK NKI (커스텀 라우팅 커널 구현) |
| 전체 학습 비용 | Trn3 (MoE 최적화 칩, 최고 가성비) |


---

## 마무리: 한눈에 보는 전체 구조

```
┌──────────────────────────────────────────────────────────────┐
│  1부: MoE란?                                                  │
│  Dense(모두 활성) → MoE(일부만 활성) = 큰 용량, 작은 비용        │
├──────────────────────────────────────────────────────────────┤
│  2부: 6가지 문제                                               │
│  ① 로드 불균형  ② 학습 불안정  ③ Expert 독립성                  │
│  ④ 메모리 문제  ⑤ 파인튜닝 낭비  ⑥ 분산 학습 복잡성             │
├──────────────────────────────────────────────────────────────┤
│  3부: 최신 해결 기술 (2025~2026)                               │
│  Loss-free 밸런싱 / CoE 체이닝 / Super Experts /              │
│  예측적 캐싱 / MoE-Sieve / Megatron Core MoE                 │
├──────────────────────────────────────────────────────────────┤
│  4부: 산업계 현황                                              │
│  DeepSeek V4 / Llama 4 / Qwen3 / Mistral Large 3            │
│  → MoE가 프론티어 AI의 기본 아키텍처로 확립                      │
├──────────────────────────────────────────────────────────────┤
│  5부: AWS 서비스                                               │
│  Bedrock(사용) → SageMaker(튜닝/학습) → Trn3/P5(인프라)       │
└──────────────────────────────────────────────────────────────┘
```

---

## 참고 자료

### 주요 논문 (3부 기술별)

| 기술 | 논문 | 링크 |
|------|------|------|
| Auxiliary-Loss-Free Load Balancing | Auxiliary-Loss-Free Load Balancing Strategy for MoE (2024) | [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) |
| Maximum Score Routing | Maximum Score Routing For Mixture-of-Experts (2025) | [arXiv:2508.12801](https://arxiv.org/abs/2508.12801) |
| Routing-Free MoE | Routing-Free Mixture-of-Experts (2026) | [arXiv:2604.00801](https://arxiv.org/abs/2604.00801) |
| SimSMoE | SimSMoE: Solving Representational Collapse (NAACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-naacl.107/) |
| MoE RL 안정화 | Stabilizing MoE RL by Aligning Training and Inference Routers (2025) | [arXiv:2510.11370](https://arxiv.org/abs/2510.11370) |
| Soft Nearest Neighbor Loss | Mixture of Experts with Soft Nearest Neighbor Loss (2026) | [arXiv:2603.26734](https://arxiv.org/abs/2603.26734) |
| Chain-of-Experts (CoE) | Unlocking the Communication Power of MoE Models (2026) | [arXiv:2506.18945](https://arxiv.org/abs/2506.18945) |
| Predictive Expert Caching | Efficient MoE Inference via Predictive Expert Caching (2025) | [arXiv:2410.17954](https://arxiv.org/abs/2410.17954) |
| Fine-Grained Offloading | Taming Latency-Memory Trade-Off in MoE Serving (2025) | [arXiv:2502.05370](https://arxiv.org/abs/2502.05370) |
| MoE-Sieve | Routing-Guided LoRA for Efficient MoE Fine-Tuning (2026) | [arXiv:2603.24044](https://arxiv.org/abs/2603.24044) |
| L-MoE | End-to-End Training of Lightweight Mixture of LoRA Experts (2025) | [arXiv:2510.17898](https://arxiv.org/abs/2510.17898) |
| Megatron Core MoE | Scalable Training of MoE Models with Megatron Core (2026) | [arXiv:2603.07685](https://arxiv.org/abs/2603.07685) |
| FSMoE | A Flexible and Scalable Training System for Sparse MoE (2025) | [arXiv:2501.10714](https://arxiv.org/abs/2501.10714) |
| MoETuner | Optimized MoE Serving with Balanced Expert Placement (2025) | [arXiv:2502.06643](https://arxiv.org/abs/2502.06643) |

### 서베이 논문

| 제목 | 링크 |
|------|------|
| A Comprehensive Survey of Mixture-of-Experts (2025) | [arXiv:2503.07137](https://arxiv.org/abs/2503.07137) |
| MoE: From Algorithmic Foundations to Decentralized Architectures (2026) | [arXiv:2602.08019](https://arxiv.org/abs/2602.08019) |
| LLM Mixture of Experts Explained — A 2026 Field Guide | [TensorOps Blog](https://tensorops.ai/blog/what-is-mixture-of-experts-llm) |

### 산업계 기술 보고서

| 모델/제품 | 링크 |
|------|------|
| DeepSeek-V4-Pro Model Card | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| Llama 4 발표 (Meta AI) | [Meta Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Qwen3 기술 블로그 | [Qwen Blog](https://qwenlm.github.io/blog/qwen3/) |
| Mistral 3 발표 | [Mistral AI](https://mistral.ai/fr/news/mistral-3) |
| EMO: Emergent Modularity (AllenAI) | [HuggingFace Blog](https://huggingface.co/blog/allenai/emo) |

### AWS 서비스 관련 링크

| 서비스/기능 | 출시일 | 링크 |
|------|------|------|
| Bedrock DeepSeek-R1 서버리스 | 2025.03.10 | [블로그](https://aws.amazon.com/blogs/aws/deepseek-r1-now-available-as-a-fully-managed-serverless-model-in-amazon-bedrock/) |
| Bedrock Llama 4 서버리스 | 2025.04.28 | [블로그](https://aws.amazon.com/blogs/aws/llama-4-models-from-meta-now-available-in-amazon-bedrock-serverless/) |
| Bedrock Mistral Large 3 | 2025.12 | [발표](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-fully-managed-open-weight-models/) |
| SageMaker Expert Parallelism | 2024.05 | [블로그](https://aws.amazon.com/blogs/machine-learning/accelerate-mixtral-8x7b-pre-training-with-expert-parallelism-on-amazon-sagemaker/) |
| SageMaker Multi-LoRA on MoE | 2026.02.26 | [블로그](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/) |
| EC2 Trn3 UltraServer GA | 2025.12 | [발표](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/) |
| EC2 Trn2 GA | 2024.12 | [제품 페이지](https://aws.amazon.com/ec2/instance-types/trn2/) |
| Neuron SDK 2.26 (Expert Parallelism beta) | 2025.09 | [발표](https://aws.amazon.com/about-aws/whats-new/2025/09/aws-neuron-2-26-announce) |
| Neuron SDK 2.29 (NKI Library) | 2026.04 | [발표](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/) |
| Deep Learning Containers (DeepSeek on EKS) | - | [가이드](https://aws.github.io/deep-learning-containers/tutorials/vllm-samples/deepseek/eks/) |

---

*작성일: 2026년 5월 13일*
*본 문서는 2026년 5월 기준 공개된 논문 및 기술 보고서를 바탕으로 작성되었습니다.*
