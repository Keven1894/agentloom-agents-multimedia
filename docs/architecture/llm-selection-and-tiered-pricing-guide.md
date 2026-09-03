# Multimedia Knowledge Distillation: LLM Evaluation, Model Selection & Tiered Pricing Architecture

**Status**: Active Architecture Guide & Evaluation Report  
**Date**: 2026-09-02  
**Target System**: MediaLoom (`agentloom-agents-multimedia`)  
**Parent Framework**: [AgentLoom Framework (v3.0)](https://github.com/Keven1894/AgentLoom)  
**Author**: Dr. Boyuan (Keven) Guan (@Keven1894) & Envita (Builder Mode)  
**Tags**: `llm-evaluation`, `video-learning`, `deepseek-r1`, `gemini-pro`, `claude-3-7`, `tiered-pricing`, `agentloom`

---

## 1. Problem Statement: Why GPT-4o Alone is Obsolete for Video Learning

In early prototypes of multimedia knowledge pipelines, standard conversational models such as **GPT-4o** were chosen as the default distillation engine. However, in practical deployment across hours of technical screencasts, academic lectures, and paid developer tutorials, relying solely on legacy GPT-4o exhibits structural limitations:

### 1.1 The Spoken-Language Gap: Condensation vs. Reconstruction
- **Oral vs. Written Disparity**: Spoken transcripts are inherently noisy. Speakers frequently use verbal filler, colloquial digressions, rhetorical questions, unscripted anecdotes, and loose transitions.
- **Surface-Level Summarization Failure**: Standard conversational models (like GPT-4o) operate on surface text reduction—they simply condense the wording. Consequently, they frequently preserve anecdotal fluff as "key takeaways" while failing to deduce the speaker's unstated foundational principles.
- **The Need for Extended Reasoning (CoT)**: Effective video learning requires **epistemic reconstruction**—a model must "think" before writing, identifying the underlying causal framework, identifying missing premises, and structuring the insight into clean concepts (Track 2) and executable skills (Track 3).

### 1.2 The Token Economics Trap
- In a 45-minute technical lecture (yielding ~20,000 Chinese characters or ~16,000 input tokens), calling flagship closed APIs at $2.50 / 1M input and $10.00 / 1M output costs ~$0.06 – $0.09 per distillation pass.
- For users actively studying multi-chapter series (10–30 episodes), naive usage of GPT-4o incurs substantial expense while delivering inferior analytical depth compared to modern reasoning and long-context architectures.

---

## 2. Frontier Model Capabilities Evaluation for Video & Course Learning

Modern frontier models bifurcate into distinct specializations. Below is an empirical assessment of the primary model families evaluated for MediaLoom's knowledge extraction engine.

### 2.1 DeepSeek Family (DeepSeek-V3 & DeepSeek-R1)
- **Role**: *The Deep Reasoning & Cost-Efficiency Champion (★ Primary Recommendation for General & Theory Courses)*
- **Key Capabilities**:
  - **DeepSeek-R1 (Thinking Model)**: Employs large-scale reinforcement learning to produce explicit Chain-of-Thought (CoT). Before emitting the summary, R1 analyzes the lecture: *"Why did the instructor choose this example? What is the core theorem being demonstrated? What are the edge cases omitted?"* It systematically strips conversational noise and reconstructs the lecture's core logic.
  - **DeepSeek-V3**: Exceptional multilingual (especially Chinese/English) fluency, ultra-fast TTFT (Time-To-First-Token), and near-zero cost.
- **Pricing**:
  - `deepseek-v3`: Input **~$0.14 / 1M tokens**, Output **~$0.28 / 1M tokens** (~95% cheaper than GPT-4o).
  - `deepseek-r1`: Input **~$0.55 / 1M tokens**, Output **~$2.19 / 1M tokens** (~75% cheaper than GPT-4o).
- **Constraints**: Context window typically 128k (plenty for 1–2 hour single lectures, but requires chunking for 10-hour full-course series).

### 2.2 Google Gemini Family (Gemini 2.5 / 3.1 Flash & Pro)
- **Role**: *The Ultra-Long Context & Native Multimodal Powerhouse (★ Best for Full-Course Panoramic Ingestion)*
- **Key Capabilities**:
  - **1M to 2.5M Context Window**: Can ingest an entire 10-lecture course curriculum (100,000+ words) in a single unified prompt. It performs cross-chapter comparisons, tracks terminology evolution across lectures, and provides panoramic syllabus summaries.
  - **Native Audio/Video Temporal Understanding**: Unrivaled native time-series understanding, yielding high-precision timestamp anchoring.
  - **Gemini Flash Variants**: Blazing fast inference suitable for high-volume automated ingestion pipelines.
- **Pricing**:
  - `gemini-2.5-flash`: Input **~$0.075 / 1M tokens**, Output **~$0.30 / 1M tokens**.
  - `gemini-3.1-pro`: Input **~$2.00 / 1M tokens**, Output **~$12.00 / 1M tokens** (for 1M+ context).

### 2.3 Anthropic Claude Family (Claude 3.5 / 3.7 Sonnet)
- **Role**: *The Software Engineering & Code SOP Synthesis Standard (★ Best for Developer Walkthroughs & Coding Courses)*
- **Key Capabilities**:
  - **Extended Thinking & Code Precision**: In programming and DevOps screencasts, instructors frequently make typographical errors or omit boilerplate error-handling. Claude 3.7 Sonnet reliably identifies these gaps, writes robust code blocks, and formats them into formal **Track 3 Candidate Executable Skills**.
  - **Pedagogical Layout**: Generates impeccably formatted, hierarchical Markdown notes, ASCII architectural diagrams, and structured concept definitions.
- **Pricing**:
  - `claude-3-7-sonnet`: Input **~$3.00 / 1M tokens**, Output **~$15.00 / 1M tokens**.

### 2.4 OpenAI Reasoning Family (o3-mini / o1)
- **Role**: *STEM, Mathematical Proofs & Scientific Lectures*
- **Key Capabilities**: Excels at rigorous mathematical derivation, physics proofs, and formal algorithmic verification from academic conference recordings.
- **Pricing**: Input **~$1.10 / 1M tokens**, Output **~$4.40 / 1M tokens** (o3-mini).

---

## 3. The 4-Tier Scenario & Pricing Architecture

To optimize both analytical depth and token expenditure, MediaLoom adopts a 4-Tier Routing Strategy:

```
┌────────────────────────────────────────────────────────────────────────┐
│               MediaLoom 4-Tier Model Selection Architecture            │
├──────────────┬──────────────────┬──────────────────┬───────────────────┤
│ Tier         │ Engine Model     │ Est. Cost / Video│ Target Scenario   │
├──────────────┼──────────────────┼──────────────────┼───────────────────┤
│ **Tier 1**   │ DeepSeek-V3 /    │ ¥0.005 ~ ¥0.02   │ High-throughput   │
│ High-Volume  │ Gemini 2.5 Flash │ ($0.001 ~ $0.003)│ news, podcasts,   │
│ Screening    │                  │                  │ daily watchlists  │
├──────────────┼──────────────────┼──────────────────┼───────────────────┤
│ **Tier 2**   │ DeepSeek-R1 /    │ ¥0.04 ~ ¥0.12    │ Humanities, logic,│
│ Deep Reason  │ o3-mini          │ ($0.006 ~ $0.018)│ business strategy,│
│ (Core Study) │                  │                  │ cognition lectures│
├──────────────┼──────────────────┼──────────────────┼───────────────────┤
│ **Tier 3**   │ Gemini 2.5 Pro / │ ¥0.15 ~ ¥0.35    │ 10+ chapter whole │
│ Full-Course  │ Gemini 3.1 Pro   │ ($0.020 ~ $0.050)│ curriculum, large │
│ Panoramic    │                  │                  │ multi-hour series │
├──────────────┼──────────────────┼──────────────────┼───────────────────┤
│ **Tier 4**   │ Claude 3.7       │ ¥0.50 ~ ¥1.50    │ Hardcore coding,  │
│ Engineering  │ Sonnet (Thinking)│ ($0.080 ~ $0.220)│ systems arch, SOP │
│ Precision    │                  │                  │ executable skills │
└──────────────┴──────────────────┴──────────────────┴───────────────────┘
```

*Benchmark Basis*: 40-minute video lecture, 16,000 input tokens, 2,000 output tokens.

---

## 4. Integration Blueprint in MediaLoom

### 4.1 Universal OpenAI-Compatible SDK Abstraction
Because DeepSeek, OpenRouter, SiliconFlow, and modern API gateways provide 100% compliant OpenAI API wire protocols (`/chat/completions`), MediaLoom's core architecture requires no vendor lock-in.

Configuration via `.env`:
```ini
# Primary LLM API Routing
LLM_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...

# Default Active Distillation Model
DISTILLATION_MODEL=deepseek-reasoner   # DeepSeek-R1
# Alternative: deepseek-chat           # DeepSeek-V3
# Alternative: claude-3-7-sonnet       # Via OpenRouter
# Alternative: gpt-4o                  # Direct OpenAI
```

### 4.2 Web UI Model Selector Matrix
The MediaLoom One-Click Ingestion Station exposes human-friendly model presets in the UI dropdown:
1. `DeepSeek-R1 (深度思维重构 · 强烈推荐)` → For in-depth concept synthesis.
2. `DeepSeek-V3 (极速经济型)` → For cost-sensitive, high-volume video parsing.
3. `Claude 3.7 Sonnet (代码工程与SOP技能)` → For software tutorials.
4. `Gemini 2.5 Pro (超长课件全景)` → For multi-episode course packages.
5. `GPT-4o (通用基准)` → Legacy baseline.

### 4.3 Automated Cost & Model Telemetry (`cost_audit`)
Every generated proposal JSON automatically records the model used, token counts, and estimated cost:
```json
"cost_audit": {
  "model": "deepseek-reasoner",
  "input_tokens": 15420,
  "output_tokens": 1980,
  "estimated_cost_usd": 0.0128,
  "estimated_cost_cny": 0.092,
  "savings_vs_gpt4o_percent": "78.4%"
}
```

---

## 5. Summary & Action Plan

1. **Retire GPT-4o as Single Default**: Shift default reasoning to **DeepSeek-R1** for substantive intellectual learning, and **DeepSeek-V3** for high-volume screening.
2. **Deploy Multi-Provider Base URL**: Enable `LLM_BASE_URL` in MediaLoom configuration to allow seamless switching between DeepSeek, OpenRouter, Google AI Studio, and OpenAI.
3. **Persist Documentation**: Retain this guide in both `agentloom-agents-multimedia/docs/architecture/` and the workspace memory registry.
