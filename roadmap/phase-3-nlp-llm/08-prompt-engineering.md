# 3.8 — Prompt Engineering

## 1. Intuition first
LLMs are in-context learners: what you put in the prompt largely determines what you get out. Prompt engineering is the discipline of designing prompts (system messages, few-shot examples, structure, tools) that reliably steer the model.

## 2. Why the topic exists
Fine-tuning is expensive; prompting is instant. Well-designed prompts can rival or exceed fine-tuned models on many tasks.

## 3. What problem it solves
Zero-shot / few-shot task adaptation, structured outputs, reasoning, tool use, safety.

## 4. Mathematics
Very little formal math. Techniques:

### 4.1 Zero-shot vs few-shot
Provide 0 or $k$ demonstrations in the prompt. Few-shot leverages in-context learning.

### 4.2 Chain-of-Thought (CoT)
Ask the model to "think step by step" — significantly improves reasoning on math/logic tasks.

### 4.3 Self-consistency
Sample $N$ reasoning chains, majority-vote the final answers.

### 4.4 Tree of Thoughts / ReAct / Reflexion
Combine reasoning with search or self-critique.

### 4.5 System / role prompts
Set persona, constraints, tone.

### 4.6 Structured output
Ask for JSON / function call; use grammar-constrained decoding (Outlines, JSON mode, tool-call).

### 4.7 Retrieval augmentation
Insert retrieved context (see 3.9).

### 4.8 Guardrails / safety prompting
"Refuse if the user requests X"; jailbreak-resistant patterns.

## 5. Every "formula" explained
Not applicable — heuristics dominate.

## 6. Variables
System, user, assistant roles; temperature; max tokens; stop sequences.

## 7. Algorithm — designing a prompt
1. State role, task, constraints.
2. Provide context (retrieved docs, examples).
3. Specify format (JSON schema, headers, bullet list).
4. Include few-shot examples.
5. Ask for reasoning ("think step by step") if needed.
6. Iterate: eval on gold set, refine.

## 8. Simple example
Bad: "Extract entities from this text."
Good:
```
You are an NER model. Extract entities as JSON: {"person": [...], "org": [...], "loc": [...]}.
Text: "Sundar Pichai runs Google in Mountain View."
JSON:
```

## 9. Real-world example
- GPT-4 with CoT scores dramatically higher on GSM8K.
- Structured JSON extraction for downstream code with `response_format={"type": "json_object"}`.
- ReAct agents drive Cursor / Copilot's tool calls.

## 10. Diagram
```mermaid
flowchart LR
    S["System: role & constraints"] --> U["User: task + context"]
    U --> LLM["LLM"]
    LLM --> R["Structured response"]
    R -->|evaluate| Iter["Refine prompt"]
```

## 11. Implementation from scratch
No code — but treat prompts as versioned artifacts (Git + evals).

## 12. Implementation using libraries
```python
from openai import OpenAI
client = OpenAI()
res = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {"role": "system", "content": "You extract entities as JSON."},
      {"role": "user", "content": text},
    ],
    response_format={"type": "json_object"},
    temperature=0.0,
)
```

Frameworks: `LangChain`, `LlamaIndex`, `Instructor`, `Guidance`, `Outlines`, DSPy (auto prompt optimization).

## 13. Time complexity
Cost = token count × price. Longer prompts = more $$$ and higher latency.

## 14. Space complexity
Watch context window; use retrieval to shorten.

## 15. Advantages
Zero training; instant iteration; portable across models.

## 16. Disadvantages
Fragile to model updates; hard to guarantee output; can leak via prompt injection.

## 17. Interview questions
1. Zero-shot vs few-shot.
2. Chain-of-thought — why does it help?
3. What is self-consistency?
4. ReAct — how does it combine reasoning + action?
5. JSON mode vs grammar-constrained decoding.
6. What is prompt injection? Mitigations.
7. Temperature vs top-p.
8. When to fine-tune instead of prompt-engineer?
9. Tricks for long-context prompting.
10. How do you evaluate a prompt?

## 18. Common mistakes
- No system message / role.
- Ambiguous format requests.
- Trusting model output without validation.
- Not lowering temperature for deterministic tasks.

## 19. Optimization techniques
Prompt compression (LLMLingua); prompt caching (Anthropic prompt caching, OpenAI Batch); DSPy for auto-optimization; contrastive prompting.

## 20. Coding exercises
1. Write a JSON extractor prompt + validator with retries.
2. Add CoT and measure lift on a small math dataset.
3. Implement ReAct loop with a calculator tool.
4. Build a prompt evaluation harness with 20 gold cases.

## 21. Mini project
Build a customer-support triage prompt with structured JSON output.

## 22. Medium project
Compare few-shot GPT-4 vs a fine-tuned small model on a classification task; report cost vs quality.

## 23. Advanced project
Use DSPy to automatically optimize a multi-hop RAG pipeline's prompts; beat hand-crafted baseline on an eval set.

## 24. Where it is used in industry
Every LLM product. Prompts are literally the code of AI applications now.

## 25. How companies use it
- Notion AI, Slack AI, GitHub Copilot: sophisticated prompt templates + tool schemas.
- Cursor: multi-file context assembly + tool prompts.

## 26. When NOT to use it
- Very high-scale simple classification — a distilled small model is cheaper.
- Cases needing hard determinism / guarantees.
