# LLM Agent Evaluation

OmniPiano includes an LLM-based evaluation path for controllers generated from a language model rather than trained from an RL checkpoint. The workflow has two stages:

1. **Practice and synthesis**: an LLM writes an open-loop `play.py` controller, executes it in simulation, inspects diagnostics, and revises it.
2. **Frozen evaluation**: the best generated keyframe controller is evaluated with the same framework-neutral evaluator used by RL policies.

This separation keeps iterative experimentation out of the final benchmark measurement. Practice may use several attempts, while the final report is produced by a fresh rollout with a stable JSON summary.

```{note}
LLM evaluation is optional and requires an OpenAI-compatible API endpoint. It does not change the Gymnasium environment contract or the evaluation metrics used for RL policies.
```

## Two evaluation modes

| Mode | Entry point | Output | Use case |
| --- | --- | --- | --- |
| Keyframe optimization | `examples/run_keyframes.py` | frozen keyframes, `play.py`, practice trajectories, conversation log | Let the LLM synthesize and improve a controller |
| Frozen controller evaluation | `examples/run_eval.py` | `eval_summary_<policy>.json` and episode CSV | Report final F1 and benchmark metrics |

The implementation lives under `omnipiano.integrations.llm`:

- `KeyframeOptimizer` runs the practice-and-revision loop.
- `KeyframePolicy` loads the frozen action sequence as a policy.
- `LLMPolicy` adapts a direct LLM action generator to the common `predict(observation, deterministic=True)` interface.
- `KeyframeOptimizerConfig` and `KeyframeConfig` keep the environment, action space, model, and output paths explicit.

## Configure the model endpoint

The LLM integration uses the OpenAI Python SDK and supports OpenAI-compatible chat-completion endpoints. The keyframe example reads the credential from `CLAUDE_API_KEY` and currently selects a Claude-compatible router endpoint in the script. If a different provider is used, update the model, `base_url`, and environment-variable handling in the example.

```{warning}
Never commit API keys to the repository or put them directly in a tutorial command. Export the credential in the shell or use a local secret manager.
```

Inspect the available options first:

```bash
python examples/run_keyframes.py --help
python examples/run_eval.py --help
```

## Supported model families and client selection

The LLM integration currently supports three API client classes. They share
the same high-level configuration (`model`, `base_url`, `api_key`, temperature,
and token budget), but differ in the wire protocol used for each request.

| Model/API family | Typical model names | Client class | When to use |
| --- | --- | --- | --- |
| OpenAI-compatible Chat Completions | OpenAI chat models, Gemini, GLM, DeepSeek, and Grok models exposed through a Chat Completions-compatible gateway | `OpenAIClient` | Default for ordinary chat-completion models and compatible providers |
| OpenAI Responses API | `gpt-6-astra` and other models exposed through the Responses API | `OpenAIResponseClient` | Use when the endpoint requires Responses API `input`, response items, reasoning, or tool calls |
| Claude-compatible tool calling | Model names beginning with `claude`, such as `claude-opus-5` | `ClaudeAgent` | Use when a Claude gateway expects Anthropic-style top-level tool schemas and content blocks |

`ClaudeAgent` and `OpenAIResponseClient` are derived from `OpenAIClient`.
They reuse its token accounting and common chat-message interface while
translating tool schemas and message history for their target protocol.

```{important}
`KeyframeOptimizer` selects the client from the model name: names beginning
with `claude` select `ClaudeAgent`; the exact name `gpt-6-astra` selects
`OpenAIResponseClient`; every other name selects `OpenAIClient`.
```

The selection rule is equivalent to:

```python
model_name = config.model.rsplit("/", 1)[-1].lower()

if model_name.startswith("claude"):
    client_class = ClaudeAgent
elif model_name == "gpt-6-astra":
    client_class = OpenAIResponseClient
else:
    client_class = OpenAIClient
```

Therefore, Gemini, GLM, DeepSeek, and Grok models should be used through an
OpenAI-compatible Chat Completions endpoint:

```python
config = KeyframeOptimizerConfig(
    env=env,
    action_space=env.action_space,
    path="runs/llm/keyframes.json",
    model="gemini-2.5-pro",       # or a GLM/DeepSeek/Grok model name
    base_url="https://provider.example/v1",
    api_key=None,
)
```

Claude models select the Claude adapter automatically:

```python
config = KeyframeOptimizerConfig(
    env=env,
    action_space=env.action_space,
    path="runs/llm/keyframes.json",
    model="claude-opus-5",
    base_url="https://4router.net/v1",
    api_key=None,
)
```

Use `OpenAIResponseClient` only when the selected model and endpoint actually
implement the Responses API. A Chat Completions endpoint is not interchangeable
with a Responses endpoint merely because both use the OpenAI Python package.

## Stage 1: practice and freeze a controller

Choose a registered environment and let the optimizer perform a small number of simulation practices:

```bash
python examples/run_keyframes.py \
    --env OmniPiano-ClairDeLune-Clean-v0 \
    --eval-seed 0 \
    --tries 3 \
    --out-dir examples/logs/eval/llm_clairdelune
```

The environment is created with `mode="eval"`, so every practice attempt has terminal musical metrics. The optimizer provides the model with structured context, including:

- the MIDI score and control timestep;
- exact action names and physical bounds;
- hand count and static keyboard partitions;
- joint names and valid ranges;
- fingertip and key geometry; and
- the current practice trajectory and error ranges.

The generated `play.py` is executed in a separate Python process. The optimizer validates its output shape, restricts imports, clips physical action targets to valid bounds, and converts them to the environment's normalized `[-1, 1]` action space before practice.

Each practice can produce diagnostics such as F1, precision, recall, missed and wrong keys, fingertip positions, non-zero actuator targets, and token usage.

The output directory typically contains:

```text
llm_clairdelune/
├── keyframes_<env-id>.json
├── keyframes_<env-id>_conversation.json
├── play_<env-id>.py
├── attempt_001_trajectory.json
├── optimization_summary.json
└── candidate_play.py
```

Use `--resume` to continue a saved conversation and practice history. The `--tries` value is the total number of practices for a resumed run:

```bash
python examples/run_keyframes.py \
    --env OmniPiano-ClairDeLune-Clean-v0 \
    --eval-seed 0 \
    --tries 5 \
    --resume \
    --out-dir examples/logs/eval/llm_clairdelune
```

```{tip}
Start with a clean, short piece and a small `--tries` value. Once the controller is correct, use the same frozen controller for robustness or multi-seed evaluation rather than changing it during measurement.
```

## Stage 2: evaluate the frozen controller

After `run_keyframes.py` has produced the keyframe JSON, run the formal evaluator:

```bash
python examples/run_eval.py \
    --env OmniPiano-ClairDeLune-Clean-v0 \
    --eval-seed 0 \
    --num-eval-eps 1 \
    --out-dir examples/logs/eval/llm_clairdelune \
    --record
```

`run_eval.py` registers the frozen controller as `llm_keyframes`, calls the shared `evaluate_policy()` function, and writes:

```text
eval_summary_llm_keyframes.json
eval_episode_metrics_<env-id>.csv
videos/                         # when --record is supplied
```

The summary includes episode returns and terminal metrics such as `episode_task/f1`, `episode_task/key_precision`, `episode_task/key_recall`, and `episode_length`.

The same evaluator can report true task reward for robust environments. Use `--eval-noise-scale` to evaluate the frozen controller at a clean, matched, or stress perturbation level:

```bash
python examples/run_eval.py \
    --env OmniPiano-ClairDeLune-A-Gauss-P10-v0 \
    --eval-seed 0 \
    --eval-noise-scale 1.5 \
    --num-eval-eps 1 \
    --out-dir examples/logs/eval/llm_action_stress
```

Here `0.0` means clean evaluation, `1.0` matches the registered training noise magnitude, and values above `1.0` apply stronger perturbations. Keep the keyframes and evaluation seed fixed when comparing scales.

## Direct LLM policies

For experiments that generate an action from each observation rather than freezing a complete trajectory, use `LLMPolicy` with the common evaluator:

```python
from omnipiano import make
from omnipiano.integrations.eval_callback import evaluate_policy
from omnipiano.integrations.llm import LLMConfig, LLMPolicy

env_id = "OmniPiano-ClairDeLune-Clean-v0"
env = make(env_id, mode="eval", seed=0)

policy = LLMPolicy(
    LLMConfig(
        action_space=env.action_space,
        model="gpt-4o-mini",
        api_key=None,  # OpenAI SDK reads OPENAI_API_KEY
        temperature=0,
    )
)

try:
    result = evaluate_policy(
        policy, env_id, env, eval_seed=0, num_episodes=1,
    )
finally:
    env.close()
```

`LLMPolicy.predict()` flattens the observation, asks the model for exactly the required number of normalized action values, validates the returned JSON array, and clips the action to the environment bounds. This mode can be expensive because it performs an API request at every control step; keyframe optimization is usually more practical for full episodes.

## Reproducibility and reporting

Retain the following with every LLM result:

- environment ID and `eval_noise_scale`;
- evaluation seed and number of episodes;
- model name, endpoint configuration, and temperature;
- prompt/conversation JSON;
- frozen keyframes and generated `play.py`;
- practice attempt metrics and token usage; and
- final `eval_summary_llm_keyframes.json`.

Practice scores are useful for controller development, but they are not a replacement for the formal evaluation summary. Report final musical quality using the same terminal F1, precision, recall, and true-reward conventions as the cross-framework evaluator.

See [Cross-Framework Evaluation](cross_framework.md) for the common result schema and [Robust Environments](../environments/robust.md) for perturbation strength semantics.
