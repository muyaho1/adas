# Design Document: AI Agent for Behavioral Pattern Response in Children with ADHD

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Proof of Concept

---

## 1. Hypothesis & Objective

**Hypothesis:** By analyzing behavioral patterns of children with behavioral issues, an AI agent can learn to respond appropriately based on those identified patterns.

**Objective:** Evaluate, through simulation, whether an RL agent can generate appropriate clinician responses tailored to ADHD behavioral patterns in school-age children (7-12).

**POC Focus:** ADHD-related task transition difficulties — impulsivity, inattention, resistance to activity changes.

---

## 2. Target Users

| User | Role |
|---|---|
| Therapists / Clinicians | Receive intervention strategy suggestions based on the agent's learned policy |
| Researchers | Evaluate the agent's response strategies, analyze discovered behavioral patterns, publish findings |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
│                                                      │
│  ┌────────────┐    action     ┌───────────────────┐ │
│  │   RL Agent  │─────────────▶│  LLM Child        │ │
│  │   (PPO)     │              │  Environment      │ │
│  │             │◀─────────────│  (Claude Code CLI) │ │
│  └────────────┘  observation  └───────────────────┘ │
│       │            + reward          │               │
│       │                              │               │
│       ▼                              ▼               │
│  ┌────────────┐              ┌──────────────┐       │
│  │  Reward     │              │  Response     │      │
│  │  Function   │              │  Cache        │      │
│  └────────────┘              └──────────────┘       │
└─────────────────────────────────────────────────────┘
                      │
                      ▼ trained policy
┌─────────────────────────────────────────────────────┐
│                Evaluation Phase                      │
│                                                      │
│  Synthetic scenarios ──▶ Trained Agent ──▶ Metrics   │
│                                  ──▶ Strategy Analysis│
│                                  ──▶ Baseline Compare │
└─────────────────────────────────────────────────────┘
```

### 3.1 Core Components

**LLM Child Environment** — A Gymnasium-compatible wrapper around Claude Code CLI that simulates a 7-12 year old child with ADHD. Given the current scenario context and the clinician's last action, it outputs the child's next behavioral state as both natural language and a structured state vector.

**RL Agent (PPO)** — Receives the behavioral state as observation, selects an intervention action from a discrete action space. Trained using Proximal Policy Optimization via Stable-Baselines3.

**Reward Function** — Research-grounded multi-dimensional scoring function that evaluates each interaction turn.

**Response Cache** — Hash-based cache mapping (scenario + profile + history + action) to LLM responses, reducing redundant CLI calls by an estimated 60-70%.

---

## 4. LLM Backend: Claude Code Wrapper

The system uses Claude Code CLI as the LLM backend instead of the Claude API to avoid per-token costs.

### 4.1 How It Works

1. The RL training loop sends a prompt to the `ClaudeCodeBackend`
2. The backend invokes Claude Code as a subprocess with the constructed prompt
3. The response is parsed to extract both natural language output and the structured state vector
4. Responses are cached to minimize repeated calls

### 4.2 Optimizations

| Strategy | Impact |
|---|---|
| Response caching (hash-based) | ~60-70% reduction in CLI calls |
| Batch mode (collect episodes, train on cache) | Decouples training from live inference |
| Reduced episode count (~500-1,000 for POC) | Manageable with CLI throughput |
| Conversation persistence within episodes | Reuse context instead of resending full history |

### 4.3 Swappable Backend Interface

```python
class LLMBackend(ABC):
    def generate(self, prompt: str) -> str: ...

class ClaudeCodeBackend(LLMBackend):  # Default — free via subscription
    ...

class APIBackend(LLMBackend):  # Optional — direct API for speed
    ...
```

### 4.4 Hardware Requirements

The system runs comfortably on a single RTX 5090 (32GB VRAM):
- RL agent: ~16K parameter MLP, trainable on CPU
- LLM inference: handled by Claude Code (remote), zero local GPU usage
- If switching to local LLM later: Llama 3 8B (Q4) requires ~6GB VRAM

---

## 5. LLM Child Environment

### 5.1 Scenario Triggers (POC Scope: Task Transitions)

- Teacher announces transition from preferred to non-preferred activity
- Timed activity ending (e.g., recess ending)
- Unexpected schedule change
- Interruption during hyperfocus

### 5.2 Child State Vector

Extracted from the LLM response via structured output parsing:

| Dimension | Range | Description |
|---|---|---|
| `distress_level` | 0.0 - 1.0 | Emotional dysregulation intensity |
| `compliance` | 0.0 - 1.0 | Willingness to follow the transition |
| `attention` | 0.0 - 1.0 | Engagement with clinician's intervention |
| `escalation_risk` | 0.0 - 1.0 | Probability of behavioral escalation next turn |

### 5.3 Step Cycle

1. Agent sends an action (e.g., "offer 2-minute warning")
2. Environment prompts Claude Code with: scenario context + child behavioral profile + interaction history + chosen action
3. Claude Code responds with a natural language behavioral description and a structured state vector
4. Environment extracts the state vector as the observation and computes reward

### 5.4 Child Profiles

Each episode samples a child profile with varying severity (mild / moderate / severe ADHD presentation) to ensure the agent generalizes across the spectrum.

### 5.5 Episode Termination

- **Max length:** 10 interaction turns
- **Early success:** compliance > 0.8
- **Early failure:** distress > 0.9 for 3 consecutive turns

---

## 6. Action Space

12 discrete clinician actions for the POC:

| # | Action | Example |
|---|---|---|
| 0 | Transition warning | "In 2 minutes we'll switch to math" |
| 1 | Offer choice | "Do you want to finish now or in 1 minute?" |
| 2 | Labeled praise | "Great job staying focused on that drawing" |
| 3 | Visual schedule cue | Show/point to the visual schedule |
| 4 | Break offer | "Let's take a quick movement break first" |
| 5 | Empathic acknowledgment | "I can see you really want to keep doing this" |
| 6 | Redirect attention | Introduce an interesting element of the next activity |
| 7 | Countdown timer | Set a visible countdown for the transition |
| 8 | Collaborative problem-solving | "What would make this switch easier for you?" |
| 9 | Ignore/wait | Give the child space, no active intervention |
| 10 | Firm boundary | "It's time. We need to move to math now" |
| 11 | Sensory support | Offer a fidget tool or calming strategy |

Actions are abstract labels. The LLM child environment interprets each action in context, so "offer choice" plays out differently depending on the scenario and child profile.

---

## 7. Reward Function

Research-grounded, multi-dimensional reward based on:

- [Emotionally Intelligent RL (2025)](https://arxiv.org/pdf/2511.10573) — reward decomposition into engagement + emotional alignment + safety penalty
- [Conditional Learning Deficits in ADHD (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8322018/) — immediacy and frequency of reinforcement matters for ADHD
- [Multi-Objective RL Practical Guide (2022)](https://link.springer.com/article/10.1007/s10458-022-09552-y) — trade-offs between competing objectives
- [VR Reward Feedback for ADHD (JMIR 2025)](https://games.jmir.org/2025/1/e67338) — combined reward modalities yield best results

### 7.1 Per-Turn Reward

```
R(t) = R_therapeutic + R_emotional + R_safety + R_efficiency

R_therapeutic = 0.3 * delta_compliance + 0.15 * delta_attention
R_emotional   = 0.25 * (-delta_distress)
R_safety      = -1.0 * (escalation_risk > 0.8)    # hard penalty
R_efficiency  = 0.1 * (1 / turn_number)            # earlier resolution preferred
```

### 7.2 Episode Bonuses

| Condition | Bonus |
|---|---|
| Compliance > 0.8 (successful transition) | +5.0 |
| Achieved in <= 5 turns | +2.0 (efficiency) |
| Distress never exceeded 0.6 | +2.0 (gentle approach) |
| Distress decreased monotonically | +1.0 (smooth de-escalation) |
| Episode ended in failure (distress > 0.9 x3) | -5.0 |

### 7.3 Design Rationale

The weights deliberately balance compliance and wellbeing. The agent cannot learn to force compliance through distress — it must find strategies that achieve transitions while keeping distress low. The hard safety penalty (instead of soft weight) ensures the agent strongly avoids escalation, consistent with the Responsible RL literature.

All weights are configurable via `configs/default.yaml` so researchers can explore different priority trade-offs.

---

## 8. RL Agent

| Parameter | Value |
|---|---|
| Algorithm | PPO (Proximal Policy Optimization) |
| Observation | 4D state vector + one-hot scenario type + turn number |
| Action space | Discrete(12) |
| Network | MLP, 2 hidden layers, 64 units each |
| Training episodes | 500 - 1,000 (POC) |
| Library | Stable-Baselines3 |

---

## 9. Evaluation Pipeline

### 9.1 Automated Metrics (Primary)

| Metric | Description | POC Target |
|---|---|---|
| Success rate | % episodes reaching compliance > 0.8 | > 70% |
| Avg turns to resolution | Mean episode length for successes | < 6 turns |
| Avg distress peak | Highest distress in successful episodes | < 0.5 |
| Safety violation rate | % episodes triggering hard penalty | < 10% |
| Monotonic de-escalation rate | % successes with steadily decreasing distress | > 40% |

### 9.2 Strategy Analysis (Research Output)

- **Action frequency heatmap** — which actions the agent prefers in which states
- **Policy visualization** — given (distress, compliance) pairs, what the agent chooses
- **Discovered sequences** — multi-step strategies (e.g., empathic acknowledgment -> transition warning -> offer choice)
- **Severity comparison** — strategy adaptation across mild / moderate / severe ADHD profiles

### 9.3 Baseline Comparisons

| Baseline | Description |
|---|---|
| Random agent | Selects actions uniformly at random |
| Rule-based agent | Follows a fixed clinical heuristic (warn -> praise -> firm boundary) |
| Single-action agent | Always picks the best single action (ablation test) |

### 9.4 Output Artifacts

- Training curves (reward over episodes)
- Policy heatmaps
- Episode transcripts (natural language logs from child environment)
- Summary statistics CSV for further analysis

---

## 10. Project Structure

```
cap/
├── src/
│   ├── environment/
│   │   ├── child_env.py
│   │   ├── child_profiles.py
│   │   ├── scenarios.py
│   │   └── state_parser.py
│   ├── agent/
│   │   ├── ppo_agent.py
│   │   └── baselines.py
│   ├── llm/
│   │   ├── backend.py
│   │   ├── claude_code_backend.py
│   │   └── api_backend.py
│   ├── reward/
│   │   └── reward_function.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── visualize.py
│   │   └── baselines_eval.py
│   └── cache/
│       └── response_cache.py
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── 01_introduction.tex
│   │   ├── 02_related_work.tex
│   │   ├── 03_methodology.tex
│   │   ├── 04_experiments.tex
│   │   ├── 05_results.tex
│   │   ├── 06_discussion.tex
│   │   └── 07_conclusion.tex
│   ├── figures/
│   │   ├── architecture.tex
│   │   ├── training_curves/
│   │   ├── policy_heatmaps/
│   │   └── action_distributions/
│   ├── tables/
│   │   ├── action_space.tex
│   │   ├── reward_weights.tex
│   │   ├── baseline_comparison.tex
│   │   └── metrics_summary.tex
│   ├── references.bib
│   └── appendices/
│       ├── child_profiles.tex
│       ├── scenario_templates.tex
│       └── episode_transcripts.tex
├── references/
│   ├── README.md
│   ├── reward_design/
│   ├── adhd_behavioral/
│   ├── therapeutic_ai/
│   ├── datasets/
│   └── simulation/
├── data/
│   ├── profiles/
│   └── scenarios/
├── results/
│   ├── training/
│   ├── evaluation/
│   └── figures/
├── configs/
│   └── default.yaml
├── train.py
├── evaluate.py
├── export_figures.py
├── requirements.txt
└── README.md
```

---

## 11. References

### Reward Function Design

| # | Paper | Year | Source | Relevance |
|---|---|---|---|---|
| 1 | Towards Emotionally Intelligent and Responsible RL | 2025 | [arXiv:2511.10573](https://arxiv.org/pdf/2511.10573) | Reward decomposition: engagement + emotional alignment + safety penalty |
| 2 | A Practical Guide to Multi-Objective RL and Planning | 2022 | [Springer AAMAS](https://link.springer.com/article/10.1007/s10458-022-09552-y) | Multi-objective scalarization approaches for balancing competing goals |
| 3 | Tiered Reward: Designing Rewards for RL | 2024 | [RLC 2024](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_159.pdf) | Structured reward tiers for shaping agent behavior |
| 4 | A Reward Alignment Metric for RL Practitioners | 2025 | [RLC 2025](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_280.pdf) | Measuring alignment between reward signal and intended behavior |
| 5 | Reward Models in Deep RL: A Survey | 2025 | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/1199.pdf) | Comprehensive survey of reward model approaches |
| 6 | Comprehensive Overview of Reward Engineering and Shaping | 2024 | [arXiv:2408.10215](https://arxiv.org/html/2408.10215v1) | Reward shaping techniques for RL applications |

### ADHD Behavioral Research

| # | Paper | Year | Source | Relevance |
|---|---|---|---|---|
| 7 | Conditional Learning Deficits in Children with ADHD | 2021 | [J Research Child Adolescent Psychopathology](https://pmc.ncbi.nlm.nih.gov/articles/PMC8322018/) | ADHD children need immediate, frequent, response-specific reinforcement |
| 8 | Reward Feedback Mechanism in VR Serious Games for ADHD | 2025 | [JMIR Serious Games](https://games.jmir.org/2025/1/e67338) | Combined coin + verbal rewards yield best inhibitory control improvement |
| 9 | How to Improve Behavioral Parent and Teacher Training for ADHD | 2020 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7585566/) | Integrating learning and motivation science into ADHD interventions |
| 10 | Which Child Will Benefit From Behavioral Intervention for ADHD | 2021 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8404726/) | Individual reward sensitivity predicts intervention efficacy |

### Therapeutic AI & RL in Mental Health

| # | Paper | Year | Source | Relevance |
|---|---|---|---|---|
| 11 | Human Guided Empathetic AI Agent for Mental Health Support | 2025 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1389041725000178) | RL-enhanced RAG for empathetic therapeutic responses |
| 12 | Adaptive Deep RL for Personalized Interventions | 2024 | [TPMAP](https://tpmap.org/submission/index.php/tpm/article/download/982/836/2039) | Deep RL adapting to evolving psychological states |
| 13 | Can RL Effectively Prevent Depression | 2025 | [World J Psychiatry](https://f6publishing.blob.core.windows.net/4a379f57-b659-4aa5-90d2-37fc4692979a/WJP-15-106025.pdf) | RL framework for preventive mental health interventions |
| 14 | Mental Health Bot Using RL | 2024 | [IJRPR](https://ijrpr.com/uploads/V5ISSUE11/IJRPR35162.pdf) | RL-based conversational agent for mental health support |
| 15 | Exploring Digital Therapeutics for Mental Health | 2024 | [WJARR](https://wjarr.com/sites/default/files/WJARR-2024-3997.pdf) | AI-driven approaches to digital mental health therapeutics |
| 16 | RL-Based Framework for Dynamic Interventions | 2024 | [Informatica](https://www.informatica.si/index.php/informatica/article/view/10316/5979) | Dynamic adaptation of therapeutic interventions using RL |

### Simulation & Agent Architecture

| # | Paper | Year | Source | Relevance |
|---|---|---|---|---|
| 17 | Generative Agents: Interactive Simulacra of Human Behavior | 2023 | [arXiv:2304.03442](https://arxiv.org/abs/2304.03442) | LLM-based agents simulating human behavior in social contexts |
| 18 | Simulating Patient-Provider Dialogues for Training Clinical NLP | 2022 | EMNLP 2022 | LLM-based patient simulation for augmenting clinical training data |
| 19 | Automated Behavioral Coding in Behavioral Health | 2022 | IEEE J Biomed Health Informatics | Real-time NLP system for coding therapist behavior |
| 20 | How to Specify RL Objectives | 2024 | [RLC 2024](https://bradknox.net/wp-content/uploads/2024/06/2024_How_to_Specify_RL_Objectives.pdf) | Framework for specifying RL objectives in applied settings |

### Datasets

| # | Dataset | Source | Access | Relevance |
|---|---|---|---|---|
| 21 | ABCD Study (Adolescent Brain Cognitive Development) | [abcdstudy.org](https://abcdstudy.org) | NDA account required | N=11,880; includes CBCL scores for anxiety, attention, aggression |
| 22 | Child Mind Institute Healthy Brain Network | [NITRC](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network) | Open access | 5,000+ participants; ADHD, anxiety, ODD diagnoses |
| 23 | ADHD-200 Consortium | [NITRC](http://fcon_1000.projects.nitrc.org/indi/adhd200) | Open access | N=973; neuroimaging + behavioral ratings |
| 24 | NIMH Data Archive (NDA) | [nda.nih.gov](https://nda.nih.gov) | Data use agreement | Aggregates dozens of pediatric mental health studies |
| 25 | CBCL Normative Dataset (ASEBA) | [aseba.org](https://aseba.org) | Licensed researchers | Gold-standard behavioral rating scales |

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| RL over prompt-engineering | Agent genuinely discovers strategies from data; stronger research contribution |
| Claude Code CLI over API | Avoids per-token costs; leverages existing subscription |
| Discrete action space (12 actions) | Manageable for POC; interpretable for clinicians and researchers |
| Hard safety penalty vs soft weight | Prevents the agent from learning coercive strategies; aligned with Responsible RL literature |
| Response caching | Makes CLI-based training feasible by reducing redundant calls |
| Configurable reward weights | Enables researchers to explore priority trade-offs (compliance vs wellbeing) |
| Data-driven, not framework-constrained | Agent discovers what works from simulation; avoids single-theory bias |

---

## 13. Success Criteria (POC)

- [ ] End-to-end training loop completes 500+ episodes
- [ ] Trained agent achieves > 70% success rate on task transition scenarios
- [ ] Agent significantly outperforms all three baselines (random, rule-based, single-action)
- [ ] Agent learns differentiated strategies for mild vs severe ADHD profiles
- [ ] Distress peak in successful episodes averages below 0.5
- [ ] Safety violation rate below 10%
- [ ] Results are reproducible with fixed random seeds

---

## 14. Future Work (Post-POC)

- Expand to ODD and anxiety-driven behavioral patterns
- Graduate to full RL training with local LLM (Llama 3 8B) for higher throughput
- Add multi-step action sequences (action combos)
- Introduce clinician expert evaluation (Phase 2 of evaluation)
- Integrate real clinical data from ABCD Study or Healthy Brain Network
- Publish findings in a peer-reviewed venue
