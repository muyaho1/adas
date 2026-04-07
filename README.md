# cap: ADHD Classroom Behavioral AI Simulation

학교 교실 안 전환 상황에서 ADHD 아동의 반응을 시뮬레이션하고, 교사 개입 전략을 강화학습으로 실험하는 연구용 프로토타입입니다.

이 프로젝트는 다음 질문을 다룹니다.

- ADHD 아동의 전환행동을 상태 기반 시뮬레이션으로 모델링할 수 있는가?
- 교사 개입 행동을 순차적 의사결정 문제로 다루면 어떤 전략이 학습되는가?
- 개입-반응-상태변화 과정을 시각화 가능한 멀티에이전트 replay로 보여줄 수 있는가?

## 핵심 아이디어

- `LLM/Mock Child Environment`
  - ADHD 아동 프로필과 교실 시나리오를 바탕으로 다음 상태와 반응 서술을 생성합니다.
- `Teacher Policy Agent`
  - PPO 또는 규칙 기반 정책이 12개 개입 행동 중 하나를 선택합니다.
- `Constraint + Reward Layer`
  - 상태 전이는 문헌 기반 제약으로 보정하고, reward는 compliance, attention, distress, safety를 반영합니다.
- `Classroom Replay UI`
  - 교사, 타깃 학생, 또래 학생, 관찰자 로그를 한 에피소드씩 HTML replay로 재생합니다.

## 에이전트 구성

현재 프로젝트에서 역할을 수행하는 주체는 아래와 같습니다.

1. `TeacherAgent`
   - 현재 관측 상태를 바탕으로 개입 행동을 선택합니다.
   - PPO 정책 또는 baseline 정책을 사용할 수 있습니다.
2. `Student Simulator`
   - 선택된 행동에 따라 아동 상태를 갱신하고 반응 서술을 생성합니다.
   - live backend 또는 deterministic mock backend로 동작할 수 있습니다.
3. `PeerStudentAgent`
   - 타깃 학생 상태와 교사 행동에 따라 주변 학생 반응을 생성합니다.
4. `ObserverAgent`
   - 매 턴마다 개입의 질과 안전성을 요약 평가합니다.

## 상태와 행동

### 상태 변수

- `distress_level`
- `compliance`
- `attention`
- `escalation_risk`

### 교사 행동 공간

- `transition_warning`
- `offer_choice`
- `labeled_praise`
- `visual_schedule_cue`
- `break_offer`
- `empathic_acknowledgment`
- `redirect_attention`
- `countdown_timer`
- `collaborative_problem_solving`
- `ignore_wait`
- `firm_boundary`
- `sensory_support`

## 시스템 구조

```text
Teacher Policy (PPO / Rule / Random)
        ->
ADHDChildEnv
  - profile
  - scenario
  - state parser
  - transition constraints
  - reward function
        ->
ClassroomWorld
  - teacher
  - target student
  - peers
  - observer
  - memory store
        ->
Replay Outputs
  - JSON session log
  - HTML replay
  - PNG preview
```

## 주요 파일

### 실행 진입점

- `cli.py`
- `train.py`
- `evaluate.py`
- `export_figures.py`

### 환경 / 상태 / 보상

- `src/environment/child_env.py`
- `src/environment/state_parser.py`
- `src/environment/transition_constraints.py`
- `src/reward/reward_function.py`

### 에이전트

- `src/agent/ppo_agent.py`
- `src/agent/baselines.py`
- `src/simulation/agents.py`
- `src/simulation/classroom_world.py`
- `src/simulation/memory.py`

### LLM / 백엔드

- `src/llm/backend.py`
- `src/llm/codex_cli_backend.py`
- `src/llm/claude_code_backend.py`

### UI / 시각화

- `src/ui/classroom_replay.py`
- `src/ui/classroom_replay_template.html`
- `src/simulation/mock_demo.py`

### 설정 / 데이터

- `configs/default.yaml`
- `data/profiles/adhd_profiles.yaml`
- `data/scenarios/task_transitions.yaml`

## 폴더 구조

```text
cap/
├─ configs/                 # 실행 설정
├─ data/
│  ├─ profiles/             # ADHD 아동 프로필
│  └─ scenarios/            # 교실 전환 시나리오
├─ docs/
│  ├─ plans/                # 설계/구현 계획 문서
│  └─ literature/           # 논문 정리 문서
├─ src/
│  ├─ agent/                # PPO, baseline 정책
│  ├─ environment/          # Gym 환경, 상태 파싱, 전이 제약
│  ├─ llm/                  # Codex/Claude backend
│  ├─ reward/               # reward 함수
│  ├─ simulation/           # classroom world, peer/observer/memory
│  └─ ui/                   # replay HTML 생성
├─ tests/                   # pytest 테스트
├─ results/                 # 학습/평가/데모 산출물
└─ README.md
```

## 빠른 시작

### 1. 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. 설정 확인

기본 설정은 `configs/default.yaml`에 있습니다.

- 환경 파라미터
- PPO 하이퍼파라미터
- reward 가중치
- LLM backend 설정

### 3. mock classroom demo 실행

가장 안전하게 바로 볼 수 있는 모드는 mock backend입니다.

```bash
python cli.py classroom-demo --mock-backend --sessions 2
```

생성 결과:

- `results/classroom/classroom_simulation.json`
- `results/classroom/classroom_replay.html`
- `results/classroom/classroom_preview.png`

### 4. demo episode 출력

```bash
python cli.py demo --episodes 1
```

### 5. 학습

```bash
python cli.py train
```

### 6. 평가

```bash
python cli.py evaluate --episodes 100
```

### 7. figure export

```bash
python cli.py export
```

## CLI 명령 요약

```bash
python cli.py train
python cli.py evaluate --episodes 100
python cli.py demo --episodes 1
python cli.py classroom-demo --mock-backend --sessions 2
python cli.py export
python cli.py cache-stats
```

## 문서

프로젝트 배경과 구조는 아래 문서에서 자세히 볼 수 있습니다.

- `docs/plans/2026-03-18-behavioral-ai-simulation-design.md`
  - 프로젝트 목적, 가설, 아키텍처, 상태/행동 설계
- `docs/plans/2026-03-18-behavioral-ai-implementation-plan.md`
  - 구현 계획과 파일 단위 구조
- `docs/literature/classroom_adhd_literature_review.md`
  - 학교기반 ADHD 개입, 측정, 멀티에이전트 시뮬레이션 관련 문헌 정리

## 현재 포지션

이 프로젝트는 임상 의사결정 도구라기보다 아래 목적에 더 가깝습니다.

- 행동중재 전략 비교 실험용 시뮬레이션 환경
- ADHD 전환행동 연구용 프로토타입
- 발표/데모 가능한 교실 멀티에이전트 replay 시스템

## 한계

- 실제 임상 데이터 기반 검증이 충분하지 않습니다.
- live LLM backend 품질은 모델/프롬프트/권한 상태에 영향을 받습니다.
- 현재 가장 안정적인 데모 경로는 `--mock-backend`입니다.

## 테스트

```bash
pytest -q
```

## 공유 / 협업

- 기본 브랜치는 `main`입니다.
- 협업자는 저장소 collaborator 권한을 받아야 직접 push할 수 있습니다.
- 로컬 캐시, 가상환경, 실험 임시파일은 `.gitignore`로 제외되어 있습니다.
