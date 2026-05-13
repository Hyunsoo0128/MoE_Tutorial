<div align="center">

# 🧠 Mixture of Experts (MoE) 기술 해설서

**2026년 프론티어 AI를 지배하는 아키텍처 — 원리, 과제, 최신 기술, AWS 활용**

[![Last Updated](https://img.shields.io/badge/Last%20Updated-May%202026-blue)]()
[![License](https://img.shields.io/badge/License-Internal-orange)]()
[![AWS](https://img.shields.io/badge/AWS-Bedrock%20%7C%20SageMaker%20%7C%20Trainium-FF9900?logo=amazonaws)]()

*GPT-5, DeepSeek-V4, Llama 4, Qwen3, Mistral Large 3 — 2026년 프론티어 모델의 공통 아키텍처*

</div>

---

## 📋 Table of Contents

- [1부: MoE란 무엇인가](#1부-moe란-무엇인가)
- [2부: MoE의 주요 문제들](#2부-moe의-주요-문제들)
- [3부: 최신 해결 기술 (2025~2026)](#3부-각-문제별-최신-해결-기술-20252026)
- [4부: 산업계 적용 현황](#4부-산업계-적용-현황-2026년-5월-기준)
- [5부: AWS에서 MoE 활용하기](#5부-aws에서-moe-활용하기)
- [참고 자료](#참고-자료)

---

## 1부: MoE란 무엇인가

### 비유로 이해하기

**종합병원**을 떠올려 보세요.

환자(토큰)가 병원에 오면, 접수처(라우터)가 증상을 보고 적절한 전문의(Expert)에게 배정합니다. 수십 명의 전문의가 있지만, 한 환자가 모든 전문의를 만나지는 않습니다. 보통 1~2명의 전문의만 진료합니다.

> - **병원 전체 의사 수** = 전체 파라미터 수 (수조 개)
> - **한 환자가 만나는 의사 수** = 활성 파라미터 수 (수백억 개)
> - **접수처** = 라우터(Router/Gate)

반대로 **Dense 모델**은 "모든 환자가 모든 의사를 만나는 병원"입니다.

### 핵심 구성요소 3가지

```
입력 토큰 → [라우터] → Expert 2, Expert 7 선택 → 가중 합산 → 출력
                ↓
           (나머지 Expert들은 비활성)
```

| 구성요소 | 설명 |
|---------|------|
| **① Expert** | 독립된 FFN. 모델 하나에 64~256개 이상 존재. 학습 과정에서 서로 다른 패턴에 특화 |
| **② Router** | 입력 토큰을 보고 상위 K개(보통 1~2개) Expert만 활성화. `Softmax(토큰 × 가중치) → Top-K` |
| **③ Sparse Activation** | 전체 Expert 중 극소수만 활성화. 256개 중 2개 = 0.8%만 연산 참여 |

### 왜 혁신인가: 파라미터 수 ≠ 연산량

| | Dense 모델 | MoE 모델 |
|---|---|---|
| 전체 파라미터 | 70B | 671B |
| 토큰당 활성 파라미터 | 70B (전부) | 37B (5.5%) |
| 토큰당 연산량 | 매우 큼 | Dense 70B와 비슷 |
| 모델 용량(지식) | 70B 수준 | 671B 수준 |

> 💡 **결론**: MoE는 "작은 비용으로 큰 모델의 지식을 사용하는 방법"입니다.

### 흔한 오해 바로잡기

> ❌ "Expert 1은 수학 전문가, Expert 2는 코드 전문가처럼 깔끔하게 나뉜다"
>
> 실제로는 **통계적이고 창발적(emergent)**인 전문화입니다. 고차원 임베딩 공간에서의 영역 분할에 가깝습니다.

> ❌ "Expert가 많으면 추론이 느려진다"
>
> 토큰당 활성화 Expert 수는 고정(1~2개)이므로, Expert를 늘려도 **토큰당 연산량은 동일**합니다. 메모리 요구량만 증가합니다.

---

## 2부: MoE의 주요 문제들

MoE는 강력하지만, 구조적으로 **6가지 핵심 문제**를 안고 있습니다.

| # | 문제 | 핵심 원인 |
|---|------|----------|
| ① | **로드 불균형 & Expert Collapse** | 양의 피드백 루프 → 소수 Expert에 집중 → 나머지 사망 |
| ② | **학습 불안정성** | 라우터-Expert 동시 학습 간섭, RL 시 라우팅 급변 → 붕괴 |
| ③ | **Expert 독립성의 한계** | 병렬 독립 처리 → Expert 간 협력 불가 → 유효 깊이 제한 |
| ④ | **추론 시 메모리 문제** | 활성 5%인데 전체 100% 메모리 상주 필요 (671B = 1.3TB) |
| ⑤ | **파인튜닝 비효율** | 모든 Expert에 adapter → Cold Expert는 노이즈만 추가 |
| ⑥ | **분산 학습 복잡성** | All-to-all 통신 + 메모리 + 연산 3중 제약 결합 |

### 문제 1: 로드 불균형 & Expert Collapse

학습 초기에 우연히 특정 Expert가 좋은 결과를 내면, 라우터는 그 Expert에 더 많은 토큰을 보냅니다. **양의 피드백 루프**가 형성되어 결국 소수 Expert만 생존합니다.

> ⚠️ **핵심 딜레마**: 강제 균등 분배(auxiliary loss)는 Expert 전문화를 방해. "균형"과 "전문화"는 본질적으로 상충.

### 문제 2: 학습 불안정성

라우터와 Expert가 동시에 학습되면서 서로 영향을 줍니다. 특히 **강화학습(RL)** 단계에서 라우터의 불안정성이 증폭되어 학습 자체가 붕괴(catastrophic collapse)할 수 있습니다.

### 문제 3: Expert 독립성의 한계

기존 MoE에서 선택된 Expert들은 **병렬·독립적**으로 작동합니다. Expert A의 출력이 Expert B의 입력에 영향을 주지 않아, 복잡한 추론에서 Expert 간 협력이 불가능합니다.

### 문제 4: 추론 시 메모리 문제

DeepSeek-V3 (671B)를 FP16으로 로드하면 **~1.3TB 메모리**가 필요합니다. H100 GPU 1장은 80GB이므로 최소 16장이 필요하지만, 실제 활성화는 5% 미만입니다.

### 문제 5: 파인튜닝 비효율

모든 Expert에 LoRA adapter를 적용하면, 실제로 활성화되지 않는 **Cold Expert**의 adapter는 학습 데이터 부족으로 노이즈를 유발합니다.

### 문제 6: 분산 학습의 복잡성

Expert Parallelism은 **all-to-all 통신**을 유발합니다. 통신량이 GPU 수의 제곱에 비례하며, 메모리·통신·연산의 3중 제약이 서로 얽혀 있습니다.

---

## 3부: 각 문제별 최신 해결 기술 (2025~2026)

### 문제 1 해결: Auxiliary-Loss-Free Load Balancing

> 📄 DeepSeek-V3 (2024) → V4 (2026) | [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) | **현재 업계 표준**

각 Expert에 **bias term** 1개를 추가합니다. 이 bias는 gradient로 학습되지 않고 단순 규칙으로 업데이트됩니다:

- 과부하 Expert → bias 낮춤 → 라우터 점수 하락 → 토큰 감소
- 저활용 Expert → bias 높임 → 라우터 점수 상승 → 토큰 증가

메인 학습 gradient와 **완전히 분리**되어 라우터의 본업(최적 배정)을 방해하지 않습니다.

### 문제 1 해결: Maximum Score Routing

> 📄 [arXiv:2508.12801](https://arxiv.org/abs/2508.12801) (2025)

기존의 capacity factor를 제거합니다. 토큰을 점수 순으로 정렬하여 처리하므로 **token dropping(정보 손실)과 padding(GPU 낭비)을 동시에 해결**합니다.

### 문제 1 해결: Routing-Free MoE

> 📄 [arXiv:2604.00801](https://arxiv.org/abs/2604.00801) (2026)

라우터를 아예 제거하고 Expert 자체가 "이 토큰을 처리할지"를 자율 판단합니다. Expert-balancing과 token-balancing을 통합 프레임워크로 동시 최적화합니다.

---

### 문제 2 해결: SimSMoE + Router-Aware IS

> 📄 SimSMoE: [NAACL 2025](https://aclanthology.org/2025.findings-naacl.107/) | RL 안정화: [arXiv:2510.11370](https://arxiv.org/abs/2510.11370)

**SimSMoE**: Expert 쌍의 출력 코사인 유사도를 측정하고, 유사도가 높으면 페널티를 부여합니다. Expert들을 "서로 밀어내는" 효과로 각 Expert가 서로 다른 표현 공간을 담당하게 합니다.

**Router-Aware IS**: RL 학습 시 라우팅이 크게 바뀐 토큰의 importance sampling 가중치를 축소하여 gradient 분산 폭발을 방지합니다.

---

### 문제 3 해결: Chain-of-Experts (CoE)

> 📄 [arXiv:2506.18945](https://arxiv.org/abs/2506.18945) (2026) | Northwestern / Oxford

기존 MoE의 **병렬 독립 처리**를 **순차 체이닝**으로 전환합니다:

```
[기존 MoE]  토큰 X → Expert 2 → 출력 A ─┐
                   → Expert 7 → 출력 B ─┴→ 가중합 (A, B 독립)

[CoE]       토큰 X → Router① → Expert 2 → X' → Router② → Expert 7 → 최종 출력
                                                (X' 기반 선택)    (X' 입력)
```

Expert 7은 Expert 2의 출력을 받아 작업합니다. **편집자가 초고를 쓰고, 교정자가 그 초고를 다듬는 것**과 같습니다.

| 지표 | 성과 |
|------|------|
| 조합 다양성 | **823배** 증가 (C(64,4)² vs C(64,8)) |
| 메모리 절감 | **42%** (12L MoE = 4L CoE) |
| Validation Loss | **1.20 → 1.12** (동일 연산량) |

### 발견: Super Experts (ICLR 2026)

수천 개 Expert 중 **3~10개**가 극단적 활성화 outlier를 생성합니다. Qwen3-30B에서 6,144개 Expert 중 **3개만 제거해도 모델 완전 붕괴**. 프루닝/양자화 시 Super Expert를 반드시 보존해야 합니다.

---

### 문제 4 해결: Predictive Expert Caching

> 📄 [arXiv:2410.17954](https://arxiv.org/abs/2410.17954) (2025) | **GPU 메모리 93.7% 절감, 처리량 10× 향상**

레이어 N의 라우팅 패턴으로 레이어 N+1에서 활성화될 Expert를 예측하고, **비동기로 CPU→GPU 프리페치**합니다:

1. 레이어 N 연산 중 → 예측 모델이 N+1 Expert 예측
2. 비동기로 CPU→GPU 전송 (연산과 동시 진행)
3. N+1 시작 시 필요한 Expert가 이미 GPU에 존재
4. Token Scheduling: 같은 Expert 쓰는 토큰을 묶어 연속 처리

> 💡 소비자 GPU (RTX 4090 24GB)에서도 수백B 파라미터 MoE 모델 실행 가능

---

### 문제 5 해결: MoE-Sieve (Routing-Guided LoRA)

> 📄 [arXiv:2603.24044](https://arxiv.org/abs/2603.24044) (2026) | **25%의 Expert만 튜닝해도 full 성능**

| Step | 내용 |
|------|------|
| **1. Profile** | 타겟 데이터로 라우팅 프로파일링 (전체 시간의 1% 미만) |
| **2. Select** | 상위 25% Hot Expert 선택, 나머지 75% Cold Expert 제외 |
| **3. Fine-tune** | Hot Expert에만 LoRA adapter 적용, Cold Expert는 동결 |

---

### 문제 6 해결: Megatron Core MoE + FSMoE

> 📄 Megatron: [arXiv:2603.07685](https://arxiv.org/abs/2603.07685) (2026) | FSMoE: [arXiv:2501.10714](https://arxiv.org/abs/2501.10714) (2025)

**Megatron Core MoE**: 메모리·통신·연산의 결합 제약을 시스템 수준에서 모델링. Expert 배치 최적화 + 통신-연산 오버랩 + 동적 부하 분산.

**FSMoE**: MoE를 4개 모듈(Token Routing / Communication / Expert Computation / Parallelism)로 추상화하고 온라인 프로파일링으로 자동 최적화.

---

## 4부: 산업계 적용 현황 (2026년 5월 기준)

> 2026년 4월 기준, Anthropic Claude를 제외한 **모든 프론티어 모델이 MoE**.

| 모델 | 총 파라미터 | 활성 | 비율 | Expert 구성 | 핵심 혁신 |
|------|-----------|------|------|------------|----------|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 3% | 256 routed + shared | Loss-free 밸런싱, Hybrid Attention, 1M ctx |
| **Llama 4 Maverick** | ~400B | 17B | 4.3% | 128 routed + 1 shared | 네이티브 멀티모달, interleaved dense/MoE |
| **Qwen3-235B** | 235B | 22B | 9.4% | 128 pool, top-8 | 다국어, 에이전트 워크플로우 |
| **Mistral Large 3** | 675B | 41B | 6% | - | 단일 8-GPU 배포, 코딩 최강 |
| **Kimi K2** | ~1T | ~32B | 3.2% | - | 네이티브 멀티모달, 에이전트 |

### 설계 철학의 분화

- **DeepSeek "극단적 세분화"**: 256개 작은 Expert + loss-free 밸런싱. 학습 비용 $5.5M
- **Llama 4 "하이브리드"**: Dense/MoE 교차 + Shared Expert. 멀티모달 네이티브
- **Qwen3 "균형잡힌 확장"**: 128개 Expert, top-8. 다국어+에이전트 균형
- **Mistral "배포 효율성"**: 단일 8-GPU 노드 목표. FP8 + 최적화 서빙

---

## 5부: AWS에서 MoE 활용하기

```
┌─────────────────────────────────────────────────────────────────┐
│                    "나는 MoE 모델을..."                           │
├─────────────────┬───────────────────┬───────────────────────────┤
│  쓰기만 하면 된다  │  파인튜닝하고 싶다  │  처음부터 학습하고 싶다     │
│                 │                   │                           │
│  Amazon Bedrock │  SageMaker AI     │  SageMaker + Trn3/P5     │
│  (API 호출)     │  + Multi-LoRA     │  + Expert Parallelism    │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### Amazon Bedrock: 서버리스 MoE 추론

| 모델 | Bedrock 출시일 | 파라미터 | 용도 |
|------|-------------|---------|------|
| [DeepSeek-R1](https://aws.amazon.com/bedrock/deepseek/) | 2025.03.10 | 671B / 37B active | 추론, 수학, 코딩 |
| [Llama 4 Maverick/Scout](https://aws.amazon.com/bedrock/meta/) | 2025.04.28 | 400B / 17B active | 멀티모달, 장문맥 |
| [Mistral Large 3](https://aws.amazon.com/bedrock/mistral/) | 2025.12 | 675B / 41B active | 코딩, 256K 컨텍스트 |
| Mixtral 8x7B | 2024 | 46.7B / 12.9B active | 비용 효율 범용 |

### SageMaker: Multi-LoRA on MoE

> 📅 2026.02 출시 | [AWS Blog](https://aws.amazon.com/blogs/machine-learning/efficiently-serve-dozens-of-fine-tuned-models-with-vllm-on-amazon-sagemaker-ai-and-amazon-bedrock/) | [vLLM Blog](https://vllm.ai/blog/multi-lora)

- 1개 MoE 기본 모델에 여러 LoRA adapter 동시 서빙
- 5개 고객 모델을 **단일 GPU**에서 처리 (활용률 10% → 80%+)
- Expert별 선택적 adapter 적용 가능 (MoE-Sieve 호환)

### SageMaker: Expert Parallelism (SMP v2)

> 📅 2024.05 출시 | [문서](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-expert-parallelism.html) | [Mixtral 가이드](https://aws.amazon.com/blogs/machine-learning/accelerate-mixtral-8x7b-pre-training-with-expert-parallelism-on-amazon-sagemaker/)

| 병렬화 기법 | 역할 |
|---|---|
| **Expert Parallelism** | Expert를 GPU 간 분산 (256개 → 32 GPU) |
| **Tensor Parallelism** | 레이어 내부를 GPU 간 분할 |
| **Pipeline Parallelism** | 레이어를 순차적으로 분할 |
| **Data Parallelism** | 데이터를 GPU 간 분할 |

### Trn3 UltraServer: MoE 전용 칩

> 📅 2025.12 GA | [제품 페이지](https://aws.amazon.com/ec2/instance-types/trn3/) | [발표](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-ec2-trn3-ultraservers/)

| 지표 | 수치 |
|------|------|
| vs Trn2 성능 | **4.4×** |
| vs Trn2 메모리 대역폭 | **3.9×** |
| vs GPU(P5e) 가성비 | **30-40% 향상** |
| 스케일 | 144 Trainium3 칩, 362 FP8 PFLOPS |

공식 타겟: *"Mixture-of-Experts, reinforcement learning, reasoning, and long-context architectures"*

### Neuron SDK: MoE 소프트웨어 스택

| 버전 | 출시일 | MoE 기능 | 링크 |
|------|-------|---------|------|
| **2.26** | 2025.09 | Expert Parallelism (beta), NKI API | [발표](https://aws.amazon.com/about-aws/whats-new/2025/09/aws-neuron-2-26-announce) |
| **2.29** | 2026.04 | NKI Standard Library, 커스텀 MoE 커널 | [발표](https://aws.amazon.com/about-aws/whats-new/2026/04/announcing-neuron-2-29/) |

### MoE 과제 ↔ AWS 서비스 매핑

| MoE 과제 | AWS 해결 방안 |
|----------|-------------|
| 추론 시 메모리 문제 | **Bedrock** (서버리스 추상화) + Inf2 |
| 파인튜닝 비효율 | **SageMaker Multi-LoRA** (선택적 adapter) |
| 분산 학습 복잡성 | **SageMaker SMP v2** + EFA (400Gbps) |
| 로드 불균형 | **Neuron SDK NKI** (커스텀 라우팅 커널) |
| 전체 학습 비용 | **Trn3** (30-40% 가성비 향상) |

---

## 참고 자료

### 주요 논문

| 기술 | 논문 | 링크 |
|------|------|------|
| Auxiliary-Loss-Free | Load Balancing Strategy for MoE (2024) | [arXiv:2408.15664](https://arxiv.org/abs/2408.15664) |
| Maximum Score Routing | Maximum Score Routing for MoE (2025) | [arXiv:2508.12801](https://arxiv.org/abs/2508.12801) |
| Routing-Free MoE | Routing-Free Mixture-of-Experts (2026) | [arXiv:2604.00801](https://arxiv.org/abs/2604.00801) |
| SimSMoE | Solving Representational Collapse (NAACL 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-naacl.107/) |
| MoE RL 안정화 | Stabilizing MoE RL (2025) | [arXiv:2510.11370](https://arxiv.org/abs/2510.11370) |
| Chain-of-Experts | Unlocking Communication Power (2026) | [arXiv:2506.18945](https://arxiv.org/abs/2506.18945) |
| Predictive Caching | Efficient MoE Inference (2025) | [arXiv:2410.17954](https://arxiv.org/abs/2410.17954) |
| Fine-Grained Offloading | Taming Latency-Memory Trade-Off (2025) | [arXiv:2502.05370](https://arxiv.org/abs/2502.05370) |
| MoE-Sieve | Routing-Guided LoRA (2026) | [arXiv:2603.24044](https://arxiv.org/abs/2603.24044) |
| L-MoE | Lightweight Mixture of LoRA Experts (2025) | [arXiv:2510.17898](https://arxiv.org/abs/2510.17898) |
| Megatron Core MoE | Scalable Training of MoE (2026) | [arXiv:2603.07685](https://arxiv.org/abs/2603.07685) |
| FSMoE | Flexible and Scalable MoE Training (2025) | [arXiv:2501.10714](https://arxiv.org/abs/2501.10714) |
| MoETuner | Balanced Expert Placement (2025) | [arXiv:2502.06643](https://arxiv.org/abs/2502.06643) |

### 서베이

| 제목 | 링크 |
|------|------|
| A Comprehensive Survey of Mixture-of-Experts (2025) | [arXiv:2503.07137](https://arxiv.org/abs/2503.07137) |
| MoE: Algorithmic Foundations to Decentralized Architectures (2026) | [arXiv:2602.08019](https://arxiv.org/abs/2602.08019) |
| LLM Mixture of Experts — A 2026 Field Guide | [TensorOps](https://tensorops.ai/blog/what-is-mixture-of-experts-llm) |

### 산업계 모델

| 모델 | 링크 |
|------|------|
| DeepSeek-V4-Pro | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) |
| Llama 4 | [Meta Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |
| Qwen3 | [Qwen Blog](https://qwenlm.github.io/blog/qwen3/) |
| Mistral 3 | [Mistral AI](https://mistral.ai/fr/news/mistral-3) |
| EMO (AllenAI) | [HuggingFace Blog](https://huggingface.co/blog/allenai/emo) |

---

<div align="center">

## Author

**Hyunsoo Kim, Ph.D.**

Senior GTM Specialist Solutions Architect — GenAI, Amazon Web Services

Ph.D., The University of Tokyo · Former AI/ML Researcher — Samsung Electronics, POSCO

[Google Scholar](https://scholar.google.com/citations?user=n09wpHYAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/hyunsoo-kim-ph-d-297b46186)

---

*Last updated: May 13, 2026*

</div>
