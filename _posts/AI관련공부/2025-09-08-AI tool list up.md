---
layout: post
title: "AI tool list up"
subtitle: AI-Tool
gh-repo: james-hyeok-kim/james-hyeok-kim.github.io.git
gh-badge: [star, fork, follow]
tags: [AI Tool]
comments: true
mathjax: true
author: James Kim
---

1. 연구기획 GPT/Gemini Prompt

```
나는 [A:나의 관심분야]를 연구하는 학자다
[B:최근 부상하는 기술]와 나의 연구분야를 융합하여,
컴퓨터 학회(NuerIPS, AAAI)에 발표할 만한 독창적 연구주제 5가지를
구체적인 연구질문 형태로 제안해줘
```

```
최근 3년간 발표된 [특정 알고리즘 또는 모델]들의 주요 구조와 성능적 한계점(예:연산량, 메모리 문제)을 요약해줘
그리고 이들을 종합하여 아직 해결되지 않은 기술적 갭과 이를 개선하기 위한 새로운 아키텍처 아이디어를 3가지 제시해줘
```

```
아래 연구 아이디어를 바탕으로, 측정 가능한 가설(measurable hypothesis)을 포함한 핵심 연구질문 1개와 하위 연구 질문 2개를 만들어줘
그리고 이 질문들에 대해 제기될 수 있는 잠재적 반론이나 기술적 맹점은 무엇인지 지적해줘
[연구아이디어 : 양자 강화학습을 이용한 양자회로 최적화
```

```
[방금 만든 구체적 연구 질문]을 검증하기 위한 실험 설계를 제안해줘
필요한 벤치마크 데이터셋은 무엇인가
비교대상으로 삼아야할 베이스라인 알고리즘은?
제안하는 모델으 ㅣ우수성을 입증하기 위한 평가지표는 무엇이 적절한가
```

```
위 논의된 연구질문과 실험 설계를 바탕으로, 컴퓨터 학회(ICML, CVRP) 제출용 논문의 표준적인 목차를 작성해줘
특히 Introduction섹션에서 연구의 동기와 기여를 강조하는 부분을 어떻게 서술하면 좋을지 예시를 보여줘
```

### 연구 질문에 답하는 AI연구 조교 - Elicit
* 질문을 던지면 관련내용을 표형태로 구조화하여 요약
* 연구방법, 데이터셋, 결과를 한눈에 비교분석 할 때, 강력
<img width="726" height="772" alt="image" src="https://github.com/user-attachments/assets/883cb846-a6a3-48fe-ba3f-1022cb5bc5e4" />

### Concensus
* Yes/No 질문에 대한 학계의 합의
<img width="734" height="779" alt="image" src="https://github.com/user-attachments/assets/c3be76f3-2b51-4e4b-8426-7353c77cbb86" />

### SciSpace
* PDF 업로드하면 논문의 어느부분이든 하이라이트하고 설명하기 가능
* 수학적 증명이나, 복잡한 실험 방법론 정독할때, 튜터 처럼 가능
<img width="715" height="769" alt="image" src="https://github.com/user-attachments/assets/f88bf62d-e642-4d97-af9c-365070a7f4be" />


### Explainpaper 
* 논문 구절 전문 해설가
* SciSpace보다 더 간단한 인터페이스
<img width="726" height="762" alt="image" src="https://github.com/user-attachments/assets/d9dc97e5-ac92-44ce-8e68-5130279ea6e0" />


### Semantic Scholar
* 스마트 논문 프로파일러
* TL;DR기능 등 논문의 핵심(표, 그림, 비디오)을 자동으로 추출
* 많은 논문을 훑어 볼때 이점
<img width="717" height="785" alt="image" src="https://github.com/user-attachments/assets/224bea39-4c17-4460-9264-a683e2831ae8" />


### Scite.ai
* 인용 맥락 분석기
* 특정 논문이 후속 연구에서 어떻게 인용되었는지, 반박되었는지를 분석
* 논문의 학술적 영향력과 논쟁을 파악하여 비판적 읽기 가능
<img width="733" height="782" alt="image" src="https://github.com/user-attachments/assets/cf2f4d94-a780-47cb-acca-f594e596bd5a" />


### Connected Papers / ResearchRabbit
* 논문 관계 시각화 도구
* Connected Papers
<img width="737" height="787" alt="image" src="https://github.com/user-attachments/assets/434eb43e-d739-4d96-a1cb-20525ee500ae" />
* ResearchRabbit
<img width="732" height="781" alt="image" src="https://github.com/user-attachments/assets/f6a5e36d-2883-4295-9e3c-3930f3371a32" />

### Cursor 
* AI기능 중심의 코드 에디터
<img width="735" height="772" alt="image" src="https://github.com/user-attachments/assets/2ad7a39c-9c56-4dd2-bdbb-09efd6423427" />

### Windsurf.ai
* 속도와 편의성 중심의 AI 코드 에디터
<img width="735" height="772" alt="image" src="https://github.com/user-attachments/assets/cb2db816-f871-4278-a700-5420c7b62900" />

### Cline
* 개발 워크플로우 자동화, AI개발 에이전트
<img width="736" height="784" alt="image" src="https://github.com/user-attachments/assets/6cd4b5d0-ce1c-44cc-bcbf-ffbf52b7215d" />

### Genspark
* 연구 아이디어를 위한 슈퍼 에이전트
* Top-Down 방식의 탐색
* 주제별 핵심 자료 제공
* 여러 하위 주제간의 관계 파악 및 연구 아이디어
* PPT 생성 AI
<img width="732" height="780" alt="image" src="https://github.com/user-attachments/assets/0e71f073-5d71-41d2-ae09-33a3a819a723" />


### AlphaXiv
<img width="732" height="780" alt="image" src="https://github.com/user-attachments/assets/ae8fe660-010d-4fef-9ea4-96fe942191ed" />

### ResearchTrend.AI
<img width="731" height="764" alt="image" src="https://github.com/user-attachments/assets/8c9d8059-e777-4a22-8fa3-9eba6abe2c1c" />

### Gamma
* PPT생성 AI
<img width="913" height="545" alt="image" src="https://github.com/user-attachments/assets/4965c10a-089e-486e-a0a4-a67b4f906d0c" />
