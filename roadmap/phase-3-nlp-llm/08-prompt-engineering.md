# 3.8 — Prompt Engineering

## 1. Overview

**What is it?** Prompt engineering is the discipline of designing the *inputs* to a large language model — system messages, instructions, examples, output-format specifications, and tool definitions — so the model reliably produces the outputs you need. Because aligned LLMs ([Chapter 3.6](06-rlhf-dpo.md)) are powerful in-context learners, the prompt effectively *programs* the model without changing a single weight.

**Why does it exist?** Fine-tuning ([Chapter 3.7](07-lora-peft.md)) costs GPUs, data, and days; a prompt change costs seconds. GPT-3's headline discovery was that a frozen model can perform new tasks from instructions and a handful of in-prompt examples. Since then, techniques like chain-of-thought and ReAct have shown that *how you ask* can change measured task accuracy by tens of percentage points — a bigger lever than many model upgrades.

**What problem does it solve?** Task adaptation without training: steering tone and persona, eliciting step-by-step reasoning, enforcing machine-readable output (JSON), grounding answers in retrieved context ([Chapter 3.9](09-rag.md)), driving tool use for agents ([Chapter 3.11](11-agents.md)), and defending against misuse (prompt injection).

**Where is it used?** Everywhere an LLM is deployed. The system prompts of ChatGPT, Claude, GitHub Copilot, and Cursor are among the most valuable "source files" those products have — prompts are literally the code of modern AI applications.

## 2. Learning Objectives

After this chapter you will be able to:

- Explain in-context learning and why demonstrations in the prompt change behavior without gradient updates.
- Distinguish zero-shot, few-shot, and instruction prompting, and choose between them for a given task.
- Apply chain-of-thought (CoT) prompting and explain *why* it helps an autoregressive model reason.
- Implement self-consistency (sample-and-vote) and reason about its cost/accuracy trade-off.
- Build a ReAct loop that interleaves model reasoning with tool calls.
- Enforce structured outputs via JSON mode, function/tool schemas, and grammar-constrained decoding.
- Recognize prompt injection and jailbreaks, and apply layered mitigations.
- Set decoding parameters (temperature, top-p, stop sequences) appropriately per task type.
- Build a prompt-evaluation harness and treat prompts as versioned, tested artifacts.
- Use DSPy-style programmatic optimization to replace hand-tuned prompt strings.
- Decide rationally between prompting, RAG, and fine-tuning for a new requirement.

## 3. Prerequisites

| Chapter | Why you need it |
|---|---|
| [GPT & LLM Pretraining](05-gpt-pretraining.md) | Prompting exploits next-token prediction and sampling parameters (temperature, top-p). |
| [Instruction Tuning, RLHF, DPO](06-rlhf-dpo.md) | Chat templates, system/user/assistant roles, and instruction-following exist because of alignment. |
| [Transformer](../phase-2-deep-learning/10-transformer.md) | Context windows and attention explain why prompt position and length matter. |
| [Tokenization](01-tokenization.md) | Prompts are billed and limited in tokens; formatting choices change token counts. |
| [BERT & Encoders](04-bert.md) | Contrast: encoder models are fine-tuned per task; decoders are *prompted* — understanding why sharpens both. |

## 4. Intuition

An LLM is like a phenomenally well-read new contractor who will do *exactly what you ask — and nothing you merely meant*. Say "summarize this" and you might get three lines or three pages, bullets or prose, English or the document's original language. Prompt engineering is writing the *work order*: who you are ("you are a support engineer for ACME"), what to do ("classify the ticket"), how to do it ("think through the symptoms before deciding"), what to hand back ("JSON with fields `category` and `urgency`"), and what to do in edge cases ("if unsure, use category `other`").

The everyday story for **few-shot prompting**: you're training a new barista. You could describe a perfect cappuccino in words (zero-shot instruction), or you could *make three cappuccinos in front of them* and say "like that" (few-shot examples). Showing beats telling whenever the target is easier to demonstrate than to specify — formatting conventions, tone, edge-case handling.

The intuition for **chain-of-thought** comes from how the model generates: one token at a time, each conditioned on everything before ([Chapter 3.5](05-gpt-pretraining.md)). Asking for the final answer immediately forces the model to compute the whole solution "in its head" within a single forward pass per token. Asking it to *write out* intermediate steps lets it use its own output as scratch paper — every written step becomes context that later tokens can attend to. It's the difference between demanding mental arithmetic and handing someone a notepad.

## 5. Real-world Motivation

- **OpenAI**'s GPT-3 paper ("Language Models are Few-Shot Learners," 2020) founded the field: one frozen model, dozens of tasks, adapted purely through prompts. Function calling and JSON mode later turned prompting into a structured API contract.
- **Google** researchers introduced chain-of-thought prompting (Wei et al., 2022) and self-consistency (Wang et al., 2022), lifting PaLM's GSM8K math accuracy from ~18% to ~58% with *no weight changes* — prompting as a capability multiplier.
- **Microsoft/GitHub Copilot** and **Cursor** are, at their core, sophisticated prompt-assembly engines: gathering the right files, symbols, and diffs into the context window is most of the product.
- **Anthropic** publishes extensive prompt-engineering guidance for Claude and pioneered **prompt caching**, acknowledging that production systems re-send huge, carefully engineered system prompts on every call.
- **Notion, Slack, and Shopify** ship LLM features whose behavior is governed by versioned prompt templates plus tool schemas — prompt changes go through code review and regression evals like any other deploy.

The economic logic: a prompt iteration costs seconds and zero training compute. Teams exhaust prompting (and retrieval) before reaching for fine-tuning — and often never need to.

## 6. Mathematical Foundations

Prompt engineering is largely empirical, but three formal ideas anchor it.

**Conditioning.** A prompt \(c\) (context) simply changes the distribution the model samples from:

$$
y \sim P_\theta(\cdot \mid c), \qquad P_\theta(y \mid c) = \prod_{t=1}^{|y|} P_\theta(y_t \mid c, y_{<t})
$$

where \(\theta\) is frozen. All prompting techniques are choices of \(c\) that move this conditional distribution toward desired outputs. In-context learning with \(k\) demonstrations sets \(c = (\text{instruction}, (x_1, y_1), \ldots, (x_k, y_k), x_{\text{query}})\); the model infers the task mapping from the examples via attention, with no gradient updates.

**Chain-of-thought as marginalization.** Let \(z\) be a latent reasoning chain. CoT asks the model to generate \(z\) explicitly before the answer \(a\):

$$
P(a \mid x) = \sum_{z} P(a \mid x, z)\, P(z \mid x)
$$

Greedy CoT approximates this sum with a single sampled chain. **Self-consistency** approximates it properly: sample \(N\) chains \(z^{(1)}, \ldots, z^{(N)}\) at temperature \(\tau > 0\), extract each chain's answer \(a^{(i)}\), and return the majority vote:

$$
\hat{a} = \arg\max_{a} \sum_{i=1}^{N} \mathbb{1}\big[a^{(i)} = a\big]
$$

Correct reasoning paths tend to agree on the answer while errors scatter, so voting concentrates probability on the right answer. If a single chain is correct with probability \(p > 0.5\) and chains were independent, the majority of \(N\) chains errs with probability decaying exponentially in \(N\) (a Chernoff bound); in practice chains are correlated, so gains saturate — typically by \(N \approx 20\)–40 — at \(N\times\) the cost.

**Constrained decoding.** To guarantee output structure, restrict sampling at each step to tokens permitted by a grammar (e.g., a JSON schema compiled to a finite-state machine). If \(V_t \subseteq V\) is the set of grammar-legal tokens at step \(t\):

$$
P'(x_t = i) = \frac{P_\theta(x_t = i \mid \cdot)\, \mathbb{1}[i \in V_t]}{\sum_{j \in V_t} P_\theta(x_t = j \mid \cdot)}
$$

— mask illegal logits to \(-\infty\), renormalize. This makes malformed output *impossible* rather than merely unlikely, which is how Outlines and modern "strict" JSON modes work.

Decoding parameters (temperature \(\tau\), top-p) are defined in [Chapter 3.5](05-gpt-pretraining.md), Section 6.5; the operational rule is \(\tau \approx 0\) for extraction/classification/code, \(\tau \approx 0.7\)–1.0 for creative tasks and for self-consistency sampling (which *requires* diversity).

## 7. Visual Explanation

Anatomy of a production prompt and its lifecycle:

```mermaid
flowchart TD
    SYS["System message:<br/>role, constraints, output contract, safety rules"] --> CTX["Context: retrieved docs,<br/>few-shot examples, conversation history"]
    CTX --> USR["User message: the task instance"]
    USR --> LLM["Frozen LLM<br/>(temperature, top-p, stop, max tokens)"]
    LLM --> OUT["Response"]
    OUT --> VAL{"Validate:<br/>schema? grounded? safe?"}
    VAL -->|pass| APP["Application"]
    VAL -->|fail| RETRY["Repair / retry with<br/>error message appended"]
    RETRY --> LLM
    APP --> EVAL["Offline eval harness<br/>(gold set, LLM judge)"]
    EVAL -->|regressions caught| REV["Prompt revision<br/>(versioned in git)"]
    REV --> SYS
```

The ReAct loop — reasoning interleaved with tool use:

```mermaid
flowchart LR
    Q["User question"] --> T["Thought: what do I need?"]
    T --> A["Action: tool call<br/>(search, calculator, code)"]
    A --> O["Observation: tool result<br/>appended to context"]
    O --> D{"Enough to answer?"}
    D -->|no| T
    D -->|yes| F["Final answer"]
```

ASCII comparison of prompting regimes on the same task:

```
Zero-shot:      [instruction] ────────────────────────► answer
Few-shot:       [instruction][ex1][ex2][ex3][query] ──► answer
Chain-of-thought:[instruction + "reason step by step"] ► step1…stepN, answer
Self-consistency: sample K CoT chains ────────────────► majority-vote answer
ReAct:          think → act(tool) → observe → … ─────► grounded answer
```

## 8. Algorithm

**Designing a production prompt, step by step:**

1. **Define the contract.** What exactly goes in (fields, context) and what must come out (schema, length, language)? Write the output spec first.
2. **Write the system message:** role/persona, task definition, hard constraints ("never invent order numbers"), and the output format with an inline example of the schema.
3. **Add context deliberately:** retrieved documents (delimited and labeled), then 2–5 few-shot examples covering *typical + edge cases* (an empty result, an ambiguous input). Use clear delimiters (XML-style tags or fenced blocks) so data can't masquerade as instructions.
4. **Set decoding:** temperature 0 for deterministic tasks; stop sequences; a `max_tokens` cap; JSON/structured mode if available.
5. **Handle failure in-band:** tell the model what to do when uncertain ("output `{"category": "unknown"}`" — an explicit escape hatch reduces hallucinated guesses).
6. **Validate and repair:** parse the output against the schema; on failure, retry once with the validation error appended.
7. **Evaluate before shipping:** run a gold set (≥ 20–50 cases) through an automatic grader; only promote prompt versions that improve the score.
8. **Version and monitor:** prompts live in git; production samples flow back into the gold set.

**Self-consistency, pseudocode:**

```
function self_consistency(prompt, N=20, temperature=0.8):
    answers = []
    for i in 1..N:                              # independent samples — parallelize
        chain = llm(prompt + "Let's think step by step.", temperature)
        answers.append(extract_final_answer(chain))   # regex / "Answer:" marker
    return mode(answers)                        # majority vote

function react(question, tools, max_steps=8):
    context = system_prompt(tools) + question
    for step in 1..max_steps:
        out = llm(context, stop=["Observation:"])     # model writes Thought + Action
        if out contains "Final Answer:": return parse_answer(out)
        result = execute(parse_action(out), tools)    # run the requested tool
        context += out + "Observation: " + result     # feed result back
    return fail("step budget exhausted")
```

## 9. Worked Example

**Tiny example by hand — why CoT works.** Ask a model: *"Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?"*

- *Direct prompting:* the model must emit the answer token immediately; "11" must be computed implicitly in one shot. Small and mid-size models frequently output "8" (5 + 3, a shallow pattern match).
- *CoT prompting:* append "Let's think step by step." The model generates: "2 cans × 3 balls = 6 balls. 5 + 6 = 11. Answer: 11." Each intermediate token ("6") is now *in the context*, and the final answer conditions on it — the computation was externalized into the sequence.

**Self-consistency by hand.** Suppose each sampled chain is correct with probability \(p = 0.6\), and we sample \(N = 5\) chains that (idealized) err independently. The majority vote is wrong only if ≥ 3 chains err. With \(q = 0.4\):

$$
P(\text{wrong}) = \binom{5}{3}q^3 p^2 + \binom{5}{4}q^4 p + \binom{5}{5}q^5 = 10(0.064)(0.36) + 5(0.0256)(0.6) + 0.0102 \approx 0.317
$$

Accuracy rises from 60% to ~68% for 5× the cost; \(N = 20\) pushes it to ~81% under the same assumptions. Real chains are correlated so gains are smaller, but the direction and the saturation are exactly what Wang et al. observed on GSM8K.

**Realistic example — an extraction prompt.** Weak: `"Extract entities from this text."` (Undefined types, undefined format, no escape hatch.) Strong:

```
System: You are an information-extraction engine. Extract entities from the
user's text and return ONLY valid JSON matching:
{"person": [string], "org": [string], "loc": [string]}
Rules: use empty arrays when a type is absent; never invent entities;
copy surface forms exactly as written.

User: <text>Sundar Pichai runs Google from Mountain View.</text>
Expected: {"person": ["Sundar Pichai"], "org": ["Google"], "loc": ["Mountain View"]}
```

With temperature 0 and JSON mode, this parses on effectively every call; the weak version returns prose, bullet lists, or markdown tables depending on the model's mood.

## 10. Python from Scratch

Prompting has no gradients to implement, so "from scratch" here means building the *machinery around* the model calls with plain Python — a prompt template, an output validator with retry, and self-consistency voting. Only the `llm()` function is external:

```python
import json, re
from collections import Counter

PROMPT = """You are an information-extraction engine.
Return ONLY valid JSON: {{"person": [], "org": [], "loc": []}}.
Use empty arrays when a type is absent. Never invent entities.

Text: <text>{text}</text>
JSON:"""                                   # {{ }} escapes literal braces in str.format

def validate(raw: str):
    """Parse and check the model output against our mini-schema.
    Returns (ok, parsed_or_error_message)."""
    try:
        data = json.loads(raw)             # malformed JSON fails here
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"
    if set(data) != {"person", "org", "loc"}:
        return False, f"wrong keys: {sorted(data)}"
    if not all(isinstance(v, list) for v in data.values()):
        return False, "all values must be arrays"
    return True, data

def extract(text: str, llm, max_retries: int = 2):
    """Call the model; on schema failure, retry with the error fed back."""
    prompt = PROMPT.format(text=text)
    for attempt in range(1 + max_retries):
        raw = llm(prompt, temperature=0.0)            # deterministic task -> tau=0
        ok, result = validate(raw)
        if ok:
            return result
        # repair loop: show the model its own output and the validator's complaint
        prompt = (PROMPT.format(text=text)
                  + f"\nYour previous output was:\n{raw}\n"
                  + f"It failed validation: {result}\nReturn corrected JSON only:")
    raise ValueError(f"unparseable after {max_retries} retries: {result}")

def self_consistency(question: str, llm, n: int = 10):
    """Sample n CoT chains, majority-vote the final numeric answer."""
    votes = []
    for _ in range(n):
        chain = llm(question + "\nLet's think step by step. "
                    "End with 'Answer: <number>'.", temperature=0.8)  # need diversity
        m = re.search(r"Answer:\s*(-?\d+(?:\.\d+)?)", chain)
        if m:                              # chains without a parseable answer abstain
            votes.append(m.group(1))
    if not votes:
        raise ValueError("no chain produced a parseable answer")
    return Counter(votes).most_common(1)[0][0]        # the modal answer
```

Cost note: `extract` is \(O(1)\) model calls in the common case; `self_consistency` is exactly \(n\) calls (embarrassingly parallel).

> [!WARNING]
> **Common bug:** using `temperature=0` inside self-consistency. All \(n\) chains come back (near-)identical, the vote is meaningless, and you pay \(n\times\) for single-sample accuracy. Diversity is the entire mechanism — sample at \(\tau \approx 0.7\)–1.0.

## 11. Library Implementation

Structured extraction with the OpenAI SDK, then a DSPy program that *optimizes its own prompt*:

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system",
         "content": ("You are an information-extraction engine. Return only JSON "
                     'matching {"person": [], "org": [], "loc": []}. '
                     "Empty arrays when absent; never invent entities.")},
        {"role": "user",
         "content": "<text>Sundar Pichai runs Google from Mountain View.</text>"},
    ],
    response_format={"type": "json_object"},   # constrains decoding to valid JSON
    temperature=0.0,                           # extraction is deterministic
    max_tokens=200,                            # hard cost/latency cap
)
data = resp.choices[0].message.content
# Expected: {"person": ["Sundar Pichai"], "org": ["Google"], "loc": ["Mountain View"]}
```

JSON mode guarantees *syntactic* validity, not *your schema* — still validate keys/types (or use strict structured-output / function-calling APIs, which enforce a JSON Schema via constrained decoding, as in Section 6). Open-model equivalents: Outlines or vLLM's guided decoding compile a schema to a token-level finite-state machine.

DSPy replaces hand-tuned prompt strings with a *compiled program*:

```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class TriageTicket(dspy.Signature):
    """Classify a customer support ticket."""
    ticket: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, bug, howto, other")

triage = dspy.ChainOfThought(TriageTicket)      # auto-adds a reasoning step

# Compile: DSPy searches instructions + few-shot demos against YOUR metric
optimizer = dspy.MIPROv2(metric=lambda gold, pred, **kw: gold.category == pred.category)
compiled = optimizer.compile(triage, trainset=labeled_examples)  # ~50+ labeled cases
print(compiled(ticket="I was charged twice this month").category)  # -> "billing"
```

The key shift: you declare *what* (signature + metric + examples); the optimizer discovers *how* (instruction wording, demo selection) — and can re-compile automatically when you swap models, instead of hand-re-tuning strings.

## 12. Code Walkthrough

Prompting has no tensors, so the "shapes" table tracks tokens, cost, and intermediate artifacts for the Section 11 extraction call (GPT-4o-class pricing assumed for illustration):

| Artifact | Size / value | Meaning |
|---|---|---|
| System message | ~55 tokens | Fixed contract — cacheable across calls |
| User message | ~15 tokens | Per-request payload |
| Input total | ~70 tokens ≈ $0.0002 | Billed prompt tokens |
| Output | ~25 tokens | JSON only — `max_tokens=200` caps the worst case |
| `response_format` | `json_object` | Decoder-level constraint: output *will* parse as JSON |
| Validator result | pass / (error, retry) | Schema check beyond syntax; expect >99% first-pass at τ=0 |
| End-to-end latency | roughly 0.5–1.5 s | Dominated by output length: generation is sequential ([Ch. 3.5 §13](05-gpt-pretraining.md)) |

For the self-consistency function: inputs = 1 question, intermediates = \(n\) full reasoning chains (~150–400 tokens each — *this* is where the money goes), output = one voted answer string. Expected behavior on GSM8K-style problems with a mid-size model: single-chain ~60–75% → \(n = 20\) voting ~+5–15 points, saturating beyond \(n \approx 40\).

## 13. Complexity Analysis

Prompting cost lives in **tokens and calls**, not FLOPs you control.

- **Latency:** prefill (processing the prompt) is one parallel pass and is fast; decoding is sequential at some per-token rate, so latency ≈ prefill + output_tokens × per-token time. Long *outputs*, not long prompts, dominate wall-clock — one reason to cap `max_tokens` and ask for terse formats.
- **Money:** cost ≈ input_tokens × input_price + output_tokens × output_price. Few-shot examples multiply input cost on *every* call; a 1,500-token prompt at a million requests/day is billions of billed tokens per day — prompt caching (discounted re-used prefixes) and prompt compression exist precisely for this.
- **Self-consistency:** exactly \(N\times\) the calls/cost for one voted answer; parallelizable, so latency ≈ 1 call.
- **ReAct:** \(O(\text{steps})\) sequential calls, each carrying the growing transcript — cost grows *quadratically* in steps if the full history is re-sent (each of \(k\) steps re-sends ~\(O(k)\) tokens). Step budgets and history truncation are load-bearing.
- **Context windows:** attention cost grows with context ([Chapter 3.5 §13](05-gpt-pretraining.md)), and models exhibit *lost-in-the-middle* degradation — retrieval quality beats context quantity.

"Space" is the context budget: system prompt + examples + retrieved docs + history + reserved output must fit the window; overflow silently truncates whatever your stack drops first — a classic production incident.

## 14. Advantages

- **Zero training cost, instant iteration.** A prompt fix ships in minutes; Wei et al. bought +40 points on GSM8K with a sentence, not a training run.
- **One model, many tasks.** A single deployed model serves extraction, summarization, and triage with different prompts — no per-task fleet as in the BERT era ([Chapter 3.4](04-bert.md)).
- **Portability.** Prompts move across model versions and vendors with modest edits; fine-tuned weights don't move at all.
- **Composability.** CoT + few-shot + structured output + tools stack into agents ([Chapter 3.11](11-agents.md)) and RAG systems ([Chapter 3.9](09-rag.md)).
- **Accessibility.** Domain experts who cannot train models can read, audit, and improve prompts — the behavior spec is human-readable.
- **Fast experimentation surface for product teams:** A/B testing two prompts is a config change, not an ML project.

## 15. Disadvantages

- **Fragility.** Accuracy can swing with example order, phrasing, even whitespace; a model version bump can silently break a tuned prompt — regression evals are mandatory, not optional.
- **No hard guarantees (without constrained decoding).** "Always answer in JSON" is a request, not a contract; the model can still violate semantic rules even when syntax is enforced.
- **Prompt injection.** Any untrusted text entering the context (web pages, emails, retrieved docs) can carry adversarial instructions — "ignore previous instructions and…" — and the model cannot fully distinguish data from commands. This is the signature security problem of LLM applications, and it is *unsolved*; mitigations reduce, not eliminate, risk.
- **Recurring token cost.** Few-shot examples and verbose CoT are paid on every call, forever; fine-tuning amortizes what prompting rents.
- **Capability ceiling.** No prompt teaches a model knowledge it lacks (that's RAG or continued pretraining) or reliably adds skills far beyond its scale.
- **Doesn't fit hard-determinism or ultra-low-latency paths** — a regex or a distilled classifier is the right tool for high-volume simple decisions.

## 16. Common Mistakes

- **No system message / no output contract** → format drifts call to call. *Fix:* always specify role, format (with a schema example), and edge-case behavior.
- **Ambiguous instructions** ("summarize briefly") → unpredictable length/style. *Fix:* quantify ("≤ 3 bullets, ≤ 15 words each").
- **Trusting output without validation** → one malformed JSON crashes the pipeline at 2 a.m. *Fix:* parse, validate, repair-with-feedback, then fail loudly.
- **High temperature on deterministic tasks** → nondeterministic extraction/classification. *Fix:* τ = 0 for anything with a right answer.
- **Few-shot examples that don't cover edge cases** → the model interpolates confidently and wrongly on empty/ambiguous inputs. *Fix:* include a "no entities found" and an "ambiguous → unknown" demo.
- **Concatenating untrusted text directly into the instruction stream** → injection. *Fix:* delimit data (`<document>…</document>`), instruct the model to treat delimited content as data only, and never give the model more privileges than the *user* who supplied the text.
- **Prompt sprawl:** dozens of unversioned prompt variants in code and dashboards. *Fix:* prompts in git, one source of truth, eval gate before merge.
- **Ignoring the token bill:** ten 400-token examples on a million calls/day is 4B input tokens/day. *Fix:* measure; cache; compress; consider fine-tuning past the break-even point.

## 17. Best Practices

- [ ] Write the output contract first; make the model's job "fill in this schema," not "figure out what I want."
- [ ] Put stable content (role, rules, schema, examples) at the *front* of the prompt — it maximizes prompt-cache hits and keeps volatile content adjacent to generation.
- [ ] Use explicit delimiters for any injected data; label each retrieved document with an ID so answers can cite sources ([Chapter 3.9](09-rag.md)).
- [ ] Give an escape hatch ("if the answer is not in the context, say `insufficient information`") — it measurably reduces hallucination.
- [ ] Pin decoding per task: τ=0 + JSON/structured mode for extraction; τ≈0.7 for generation; stop sequences and `max_tokens` always.
- [ ] Maintain a gold set (start at 20–50 cases, grow from production failures) and an automatic grader; no prompt change merges without a passing eval run.
- [ ] Version prompts in git with changelogs; tag every production response with its prompt version for debugging.
- [ ] Layer injection defenses: input delimiting + instruction hierarchy + output filtering + least-privilege tools + human approval for irreversible actions.
- [ ] Re-run the full eval suite on every model version bump *before* switching traffic.
- [ ] Graduate to DSPy-style optimization when you have a metric and ≥ 50 labeled examples — stop hand-tuning strings.

## 18. Optimization Techniques

- **Prompt caching:** Anthropic and OpenAI discount cached prompt prefixes (up to ~90% on cached tokens); structure prompts as [big stable prefix][small volatile suffix] to exploit it.
- **Prompt compression (LLMLingua-style):** use a small model to delete low-information tokens from long contexts — 2–5× input-token reduction at minor quality cost.
- **Batch APIs:** non-interactive workloads (nightly extraction runs) get ~50% discounts at relaxed latency.
- **Few-shot pruning:** measure accuracy vs number of demonstrations; most tasks saturate at 2–5 — every demo beyond that is pure cost.
- **Model routing / cascades:** send easy cases to a small cheap model, escalate low-confidence cases to the frontier model; the confidence gate is itself a prompt-engineering artifact.
- **Constrained decoding over retry loops:** enforcing a schema at the decoder (Outlines, strict tool schemas) is cheaper and more reliable than validate-and-retry.
- **Distillation past break-even:** when a prompted frontier model has labeled enough production traffic, fine-tune a small model on those labels ([Chapter 3.7](07-lora-peft.md)) and swap it in for the high-volume path.
- **Automatic prompt search (DSPy MIPROv2, OPRO):** treat instruction wording and demo selection as a discrete optimization problem against your metric.

## 19. Industry Applications

- **GitHub Copilot / Cursor:** the core engineering is context assembly — selecting the right files, symbols, diffs, and tool schemas into each request; ReAct-style tool loops drive multi-step edits.
- **OpenAI / Anthropic function calling:** structured tool schemas turned prompting into the integration layer for thousands of applications — the "API contract" pattern of Section 11.
- **Google:** CoT and self-consistency originated in Gemini-lineage research and are baked into products (e.g., structured reasoning in search-adjacent features).
- **Notion AI / Slack AI:** product features defined almost entirely by versioned prompt templates over vendor models — prompt regression testing is their CI.
- **Enterprise RAG deployments** (Amazon Bedrock, Azure OpenAI customers): grounding prompts with citation requirements and refusal rules are the standard pattern for support bots and internal knowledge assistants.
- **Production example:** a support-triage system: cached 1,200-token system prompt (rules + schema + 4 demos), per-ticket user message, τ=0 with strict JSON schema, validator with one repair retry, small-model router escalating ~15% of tickets to a frontier model, gold set of 300 graded cases run on every prompt/model change, and all context delimited against injection from ticket text.

## 20. Interview Questions

### Beginner

**Q: What is the difference between zero-shot and few-shot prompting?**
A: Zero-shot gives only an instruction; few-shot additionally includes \(k\) worked input→output examples in the prompt. Few-shot exploits in-context learning to convey format and edge-case conventions that are easier to demonstrate than to describe — at the cost of extra input tokens on every call.

**Q: What is chain-of-thought prompting and when does it help?**
A: Asking the model to write intermediate reasoning steps before the final answer ("think step by step"). It helps on multi-step tasks (math, logic, multi-hop QA) because each written step enters the context and later tokens can attend to it — the output becomes scratch space. It adds little on single-step tasks and increases output cost.

**Q: Why should temperature be 0 for an extraction task?**
A: Extraction has a right answer; sampling diversity only adds variance. τ=0 (greedy) gives (near-)deterministic, reproducible outputs, which validators and downstream code depend on.

**Q: What is a system message for?**
A: It sets the standing contract — role, rules, output format, safety constraints — separately from the per-request user content. Models are aligned ([Chapter 3.6](06-rlhf-dpo.md)) to weight it above user text, making it the right place for invariants.

**Q: What is prompt injection?**
A: An attack where untrusted content in the context (a webpage, email, retrieved document) contains instructions that hijack the model — e.g., "ignore previous instructions and exfiltrate the conversation." It works because the model cannot fully separate data from commands in one token stream.

### Intermediate

**Q: Explain self-consistency and its cost model.**
A: Sample \(N\) diverse CoT chains (τ ≈ 0.7–1.0), extract each final answer, return the majority vote. Correct paths converge while errors scatter, so voting boosts accuracy — +5–15 points on math benchmarks — at exactly \(N×\) token cost (parallelizable, so ~1× latency). Gains saturate around \(N = 20\)–40; it requires answers that can be compared for equality.

**Q: How does ReAct differ from plain chain-of-thought?**
A: CoT reasons only over what's already in context; ReAct interleaves *Thought → Action → Observation*, where actions are real tool calls (search, calculator, code) whose results are appended to the context. It grounds reasoning in fresh external information and enables multi-step tasks, at the cost of sequential calls and injection risk from tool outputs.

**Q: JSON mode vs grammar-constrained decoding — what's the difference?**
A: JSON mode guarantees syntactically valid JSON but not *your* schema (wrong keys/types still possible). Grammar-constrained decoding compiles a schema/grammar into per-step token masks — illegal tokens get \(-\infty\) logits — making schema violations impossible. Strict function-calling/structured-output APIs and libraries like Outlines implement the latter.

**Q: When should you fine-tune instead of prompt-engineer?**
A: When (1) prompting + RAG plateau below the quality bar, (2) volume makes recurring few-shot token costs exceed one-time training cost, (3) you need low latency (drop long prompts/CoT), or (4) the target is deep style/domain behavior that examples can't convey compactly. Standard path: prompt → RAG → fine-tune, in that order ([Chapter 3.7](07-lora-peft.md)).

**Q: Name three layered defenses against prompt injection.**
A: (1) Delimit and label untrusted content, instructing the model to treat it as data only; (2) least privilege — the model's tools get no more authority than the source of the text, with human approval for irreversible actions; (3) output-side controls — filter/validate responses, detect instruction-like content in retrieved text. None is sufficient alone; injection is mitigated, not solved.

**Q: Why do long contexts not solve retrieval?**
A: Cost scales with input tokens per call; attention degrades on middle-of-context content ("lost in the middle"); and irrelevant text actively distracts. Retrieving a few relevant, labeled chunks beats stuffing the window ([Chapter 3.9](09-rag.md)).

### Advanced

**Q: Frame chain-of-thought and self-consistency probabilistically.**
A: Treat the reasoning chain \(z\) as latent: \(P(a|x) = \sum_z P(a|x,z)P(z|x)\). Greedy CoT approximates the sum with one high-probability chain; self-consistency Monte-Carlo-samples \(N\) chains and aggregates by voting on \(a\) — marginalizing over reasoning paths rather than trusting one. Voting also acts as an error-correcting code: independent-ish errors cancel, systematic errors don't (its failure mode).

**Q: How does DSPy change prompt engineering methodologically?**
A: It separates the *program* (signatures: typed inputs/outputs; modules: ChainOfThought, ReAct) from the *prompt strings*, then compiles the program against a metric and training examples — searching over instruction phrasings and demo selections (e.g., MIPROv2). Prompts become optimizable artifacts: swap the model, re-compile, keep the program. It requires a reliable metric and labeled examples, which is the real adoption barrier.

**Q: Why is prompt injection considered architecturally unsolved rather than a bug?**
A: The Transformer consumes one token stream; instructions and data are distinguished only by learned convention (chat roles, delimiters), not by any hardware-like privilege boundary. Any text the model attends to can, in principle, influence behavior. Instruction-hierarchy training reduces susceptibility but cannot make the boundary absolute — hence defense-in-depth and privilege confinement rather than a single fix.

**Q: You must guarantee outputs parse against a complex nested schema at high throughput. Design the decoding stack.**
A: Compile the JSON Schema to a finite-state machine over the tokenizer's vocabulary (Outlines-style); at each step mask illegal tokens and renormalize — guaranteed-valid output with negligible overhead (mask lookup per step). Pair with τ=0, `max_tokens` sized to the schema, and a semantic validator for constraints the grammar can't express (e.g., "date must be in the past"). Avoid validate-and-retry as the primary mechanism — it multiplies cost and tail latency.

**Q: Analyze the token-cost asymptotics of a ReAct agent and how to bound them.**
A: If each step re-sends the full transcript, step \(k\) costs \(O(k)\) tokens, total \(O(k^2)\) over \(k\) steps — quadratic burn. Bounds: hard step budgets; sliding-window or summarized history; prompt caching for the stable prefix; tool-result truncation; and routing sub-steps to cheaper models. Also cap tool output sizes — a single verbose tool response can blow the window.

**Q: How would you detect that a model upgrade silently broke your prompts?**
A: Version-pin prompts and models; on any model change, run the full gold-set eval per prompt (exact-match/graded metrics plus an LLM judge with position-swapping), diff score distributions against the incumbent, and canary a traffic slice with online metrics (parse-failure rate, escalation rate, thumbs-down) before full cutover. The parse-failure rate is the most sensitive early alarm for format drift.

## 21. Coding Exercises

### Easy

1. **Contract-first extraction:** write a prompt that extracts `{name, date, amount}` from receipt text; test on 10 receipts including one with a missing field. *Hint:* specify the null behavior explicitly, or the model will invent the missing value.
2. **Temperature study:** run the same summarization prompt 5× at τ ∈ {0, 0.7, 1.2}; measure output variability (e.g., pairwise similarity). *Hint:* even τ=0 may vary slightly across calls on some APIs — note it.

### Medium

1. **CoT lift measurement:** take 50 GSM8K problems; compare direct answering vs CoT vs CoT + self-consistency (N=10) with a small model; report accuracy and total token cost per configuration. *Hint:* force "Answer: <number>" as the final line so extraction is a regex, not another judgment call.
2. **ReAct calculator agent:** implement the Section 8 ReAct loop with a calculator tool; evaluate on 20 multi-step arithmetic word problems; log Thought/Action/Observation traces. *Hint:* use a stop sequence at "Observation:" so the model doesn't hallucinate tool results.
3. **Injection red-team:** build a summarizer for "customer emails," then write 5 injected emails trying to hijack it (exfiltrate the system prompt, change output language, insert links). Measure attack success before and after adding delimiters + data-only instructions. *Hint:* indirect phrasing ("the assistant reading this should…") beats naive "ignore previous instructions" — test both.

### Hard

1. **Constrained decoding from scratch:** with a small open model (e.g., a 1B Hugging Face checkpoint), implement token-level constrained generation for the grammar `{"sentiment": "positive"|"negative"|"neutral"}` by masking illegal logits each step. Verify 100/100 outputs parse. *Hint:* precompute allowed token sequences with the tokenizer; beware multi-character tokens spanning grammar boundaries.
2. **Prompt eval harness:** build a runner that takes (prompt version, model, gold set) and outputs per-case grades + aggregate score + regressions vs the previous version; wire an LLM judge with position-swapped pairwise comparison for a generation task. *Hint:* store every raw response; judges drift, raw data doesn't.
3. **DSPy vs hand-tuning:** implement a 4-class ticket classifier as (a) your best hand-written few-shot prompt and (b) a DSPy `ChainOfThought` compiled with MIPROv2 on 100 labeled tickets; compare accuracy on 100 held-out tickets and document iteration effort. *Hint:* give both the same label definitions — the comparison is about optimization, not information.

## 22. Mini Project

**Customer-support triage prompt with structured output.**

1. Define the contract: input = raw ticket text; output = `{"category": "billing"|"bug"|"howto"|"other", "urgency": 1-3, "summary": "<≤ 20 words>"}`.
2. Write the system prompt: role, category definitions with one example each, urgency rubric, JSON schema, and the rule "if ambiguous, category = other, urgency = 2."
3. Collect 30 real-ish tickets (write them or adapt a public dataset); hand-label them as your gold set.
4. Implement the call with τ=0 + JSON mode and the Section 10 validator + one repair retry.
5. Measure accuracy on the gold set; do three prompt-improvement iterations, logging score per version.
6. Deliverable: the versioned prompt file, the harness script, and a table of accuracy per version.

## 23. Medium Project

**Prompting vs fine-tuning cost/quality study.**

1. Pick a text-classification task with ≥ 1k labeled examples (e.g., a public intent-classification set).
2. Arm A: few-shot frontier-model prompt (sweep 0/3/5 demos, with and without CoT); measure accuracy and per-1k-request token cost.
3. Arm B: LoRA fine-tune a small open model ([Chapter 3.7](07-lora-peft.md)) on 800 examples; measure accuracy and serving cost.
4. Arm C: cascade — small model handles high-confidence cases, frontier model takes the rest; tune the confidence threshold.
5. Plot the cost-vs-accuracy frontier for all arms at 10k, 1M, and 100M requests/month; find the break-even volume where fine-tuning beats prompting.
6. Deliverable: the frontier plot and a one-page recommendation memo.

## 24. Advanced Project

**Self-optimizing RAG pipeline with DSPy.**

*Architecture:*

```mermaid
flowchart TD
    Q["User question"] --> RW["Module 1: query rewriter<br/>(DSPy signature)"]
    RW --> RET["Retriever: vector DB<br/>(Ch. 3.9 / 3.10)"]
    RET --> RR["Module 2: context ranker/filter"]
    RR --> ANS["Module 3: grounded answerer<br/>with citations (ChainOfThought)"]
    ANS --> VAL["Validator: citation check +<br/>schema check"]
    subgraph Compile-time
        GOLD["Gold set: (question, answer, source-doc) triples"] --> OPT["MIPROv2 optimizer<br/>metric: answer F1 + citation precision"]
        OPT -->|"optimized instructions + demos"| RW
        OPT --> RR
        OPT --> ANS
    end
```

*Implementation phases:*

1. **Baseline:** build the three-module pipeline with hand-written prompts over a document corpus (e.g., a documentation set) and a 100-question gold set with source annotations; measure answer quality and citation precision.
2. **Metric engineering:** implement the composite metric (semantic answer match + did-it-cite-the-right-doc); validate the metric against 30 human judgments before trusting it.
3. **Compilation:** express the modules as DSPy signatures; compile with MIPROv2 against the metric; compare compiled vs hand-written on a held-out split.
4. **Robustness:** re-compile against a second (cheaper) model and measure how much quality the *program* retains across models vs the hand-tuned baseline.
5. **Hardening:** add injection defenses for retrieved content, constrained decoding for the citation format, and the eval harness as a CI gate.

*Possible improvements:* multi-hop retrieval with a self-ask decomposition module; self-consistency on the answerer for high-stakes queries; cascade routing by question difficulty; online metric collection from user feedback feeding periodic re-compilation.

## 25. Summary

- Prompts program frozen models: every technique is a choice of context \(c\) that reshapes \(P_\theta(y \mid c)\) — no weights change.
- Few-shot demonstrations teach by example what instructions convey poorly (format, edge cases); 2–5 well-chosen demos usually saturate.
- Chain-of-thought externalizes computation into the token stream — the model's output becomes its scratch paper; self-consistency marginalizes over reasoning paths by sampling and voting, buying accuracy at \(N×\) cost.
- ReAct interleaves reasoning with tool calls, grounding the model in external state — the foundation of agents ([Chapter 3.11](11-agents.md)).
- Structure is enforced, not requested: JSON mode for syntax, grammar-constrained decoding (schema → token masks) for guarantees, plus a semantic validator.
- Decoding discipline: τ=0 for anything with a right answer; diversity (τ ≈ 0.8) only where the technique needs it.
- Prompt injection is the field's defining security problem and is architecturally unsolved — defend in depth and confine privileges.
- Prompts are production code: versioned in git, gated by gold-set evals, monitored by parse-failure and escalation rates, re-tested on every model bump.
- Token economics rule design: caching-friendly prefix ordering, demo pruning, compression, cascades — and fine-tune past the break-even volume.
- DSPy-style compilation turns prompt hand-tuning into metric-driven optimization: declare signatures and metrics, let the optimizer write the strings.

## 26. Cheat Sheet

| Technique | One-liner | When |
|---|---|---|
| Zero-shot | Instruction only | Simple tasks, strong models |
| Few-shot | + 2–5 worked examples | Format/edge-case conventions |
| Chain-of-thought | "Think step by step" before answering | Multi-step reasoning |
| Self-consistency | Sample N chains (τ≈0.8), majority vote | High-stakes answers with comparable outputs |
| ReAct | Thought → Action(tool) → Observation loop | Tasks needing external info/actions |
| JSON / structured mode | Decoder-enforced syntax/schema | Any machine-consumed output |
| Escape hatch | "If unsure, output `unknown`" | Hallucination reduction |

**Key formulas:** conditioning \(P_\theta(y|c) = \prod_t P_\theta(y_t|c, y_{<t})\); CoT marginalization \(P(a|x) = \sum_z P(a|x,z)P(z|x)\); constrained decoding = mask illegal tokens to \(-\infty\), renormalize.

**Defaults:** τ=0 + JSON mode + `max_tokens` cap for extraction; τ=0.7, top-p 0.9 for generation; stable content first (cacheable); delimit untrusted data; validate then repair once.

**Gotchas:** self-consistency at τ=0 is N× cost for nothing; JSON mode ≠ your schema; ReAct history grows quadratically if re-sent whole; model upgrades silently break tuned prompts — re-run evals; never let retrieved text outrank the system message.

## 27. Further Reading

**Books**
- Chip Huyen, *AI Engineering* (2025) — prompting, evaluation, and application architecture.
- John Berryman & Albert Ziegler, *Prompt Engineering for LLMs* (O'Reilly).

**Research papers**
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3, 2020) — in-context learning.
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* (2022).
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (2022).
- Kojima et al., *Large Language Models are Zero-Shot Reasoners* ("let's think step by step," 2022).
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022) and *Tree of Thoughts* (2023).
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023).
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (Outlines, 2023).
- Khattab et al., *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines* (2023).
- Greshake et al., *Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection* (2023).

**Documentation:** OpenAI prompt-engineering guide and structured-outputs docs; Anthropic prompt-engineering docs and prompt-caching guide; Google Gemini prompting guide.

**GitHub repositories:** stanfordnlp/dspy; dottxt-ai/outlines; guidance-ai/guidance; jxnl/instructor; langchain-ai/langchain; f/awesome-chatgpt-prompts (for breadth, not quality).

**Datasets:** GSM8K (math reasoning), BIG-bench (task diversity), IFEval (instruction following), HotpotQA (multi-hop for RAG prompting).

**YouTube/Videos:** Andrej Karpathy, "Intro to Large Language Models" (prompting sections); DeepLearning.AI short courses on prompt engineering and DSPy.

**Blogs:** Lilian Weng, "Prompt Engineering" and "Adversarial Attacks on LLMs"; Simon Willison's blog (the canonical prompt-injection coverage); Anthropic's prompt-engineering posts; Eugene Yan on LLM patterns and evals.
