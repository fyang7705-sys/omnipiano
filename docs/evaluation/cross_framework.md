# Cross-Framework Evaluation

OmniPiano provides a framework-neutral evaluation layer so policies trained
with different reinforcement-learning libraries can be measured with the same
environment construction, seeds, return definition, terminal metrics, and
JSON schema.

The central functions live in `omnipiano.integrations.eval_callback`:

| Component | Responsibility |
| --- | --- |
| `evaluate_policy()` | Runs deterministic episodes and aggregates returns, lengths, and terminal metrics |
| `write_eval_summary()` | Writes the result dictionary as stable, sorted JSON |
| `EvalCallback` | Schedules periodic evaluation from an external training loop |
| `postprocess_eval_csv()` | Adds authoritative training steps to episode CSV files using `evaluations.npz` |

This layer does not import SB3, TorchRL, OmniSafe, RLlib, or another training
framework. Framework-specific code only needs to adapt its actor to the small
prediction interface below.

```{note}
The evaluator standardizes rollout and reporting behavior, not checkpoint
formats. Each framework remains responsible for loading its model and exposing
the `predict()` method.
```

## Policy interface

An evaluation policy must expose:

```python
policy.predict(observation, deterministic=True)
```

The method may return either an action array or a tuple whose first element is
the action. The tuple form matches Stable-Baselines3's `(action, state)`
return value; the array form is convenient for lightweight and TorchRL
adapters.

For example, a callable NumPy actor can be adapted with a few lines:

```python
class EvaluationPolicy:
    def __init__(self, actor):
        self.actor = actor

    def predict(self, observation, deterministic=True):
        return self.actor(observation)
```

The framework adapter owns tensor conversion, device placement, recurrent
state, normalization, and checkpoint loading. `evaluate_policy()` deliberately
does not infer those framework-specific details.

```{tip}
Keep the adapter thin. A small `predict()` wrapper is easier to audit than a
second evaluation loop that silently changes seeding or return semantics.
```

## Create the evaluation environment

Evaluation environments must be created with `mode="eval"`. This enables
musical precision, recall, and F1 computation. Passing `log_dir` additionally
attaches the episode CSV logger.

```python
from pathlib import Path

from omnipiano import make

env_id = "OmniPiano-ClairDeLune-Clean-v0"
run_dir = Path("runs/cross_framework/example")
run_dir.mkdir(parents=True, exist_ok=True)

eval_env = make(
    env_id,
    mode="eval",
    seed=100,
    log_dir=str(run_dir),
)
```

For robust environments, set `eval_noise_scale` on the same registered task:

```python
eval_env = make(
    "OmniPiano-ClairDeLune-A-Gauss-P10-v0",
    mode="eval",
    eval_noise_scale=1.5,
    seed=100,
    log_dir=str(run_dir),
)
```

`0.0` gives a clean evaluation, `1.0` gives matched evaluation at the
registered training strength, and values above `1.0` are stress tests. See
[Robust Environments](../environments/robust.md) for the full perturbation
contract.

## Run a final evaluation

```python
from omnipiano.integrations.eval_callback import (
    evaluate_policy,
    write_eval_summary,
)

try:
    result = evaluate_policy(
        policy,
        env_id,
        eval_env,
        eval_seed=100,
        num_episodes=5,
    )
finally:
    eval_env.close()

result.update(
    {
        "algorithm": "MyAlgorithm",
        "seed": 0,
        "total_env_steps": 5_000_000,
        "protocol_version": "1.0",
    }
)
write_eval_summary(result, run_dir / "eval_summary.json")
```

Each episode is reset with:

```text
eval_seed + episode_index * 10,000
```

The explicit spacing makes episode seeds reproducible and avoids consuming an
unknown continuation of the training RNG stream.

## Return semantics

By default, `evaluate_policy(..., true_reward=True)` accumulates
`info["task/true_reward"]`. This is the unperturbed task reward emitted before
reward noise is applied. It is the appropriate value for final benchmark
comparison because it measures task performance rather than sensor corruption.

```{important}
Periodic model selection and final benchmark reporting answer different
questions. Periodic callbacks may rank policies by received reward; the final
benchmark should normally report true task reward plus musical F1.
```

Set `true_reward=False` only when the desired quantity is the reward actually
received by the policy:

```python
observed_result = evaluate_policy(
    policy,
    env_id,
    eval_env,
    eval_seed=100,
    num_episodes=5,
    true_reward=False,
)
```

The distinction matters only for reward-robust environments. For clean,
action-noise, observation-noise, and physical-noise tasks, the observed and
true reward definitions are otherwise aligned.

## Metrics and summary schema

At the terminal step, the evaluator collects numeric `info` entries whose keys
start with `episode_`. Typical fields include:

| Metric | Meaning |
| --- | --- |
| `episode_task/f1` | Musical key-activation F1 |
| `episode_task/key_precision` | Precision of played keys |
| `episode_task/key_recall` | Recall of target keys |
| `episode_task/sustain_f1` | Sustain-pedal F1 |
| `episode_safety/cost_total` | Accumulated safety cost |
| `episode_safety/violations` | Number of unsafe steps/events |
| `episode_task/*_reward` | Episode reward decomposition |

The returned object has this shape:

```json
{
  "env_id": "OmniPiano-ClairDeLune-Clean-v0",
  "eval_seed": 100,
  "episodes": [
    {
      "episode_index": 0,
      "episode_return": 123.4,
      "episode_length": 588,
      "metrics": {
        "episode_task/f1": 0.72
      }
    }
  ],
  "summary": {
    "return_mean": 123.4,
    "return_std": 0.0,
    "length_mean": 588.0,
    "episode_task/f1_mean": 0.72,
    "episode_task/f1_std": 0.0
  }
}
```

Every discovered episode metric receives `_mean` and `_std` summary fields.
Additional audit metadata such as algorithm name, training seed, protocol
version, checkpoint, and effective hyperparameters should be added by the
calling integration before writing the JSON.

## Protocol defaults

`BenchmarkProtocolConfig` is the shared source of cross-framework defaults:

| Field | Default | Purpose |
| --- | ---: | --- |
| `total_env_steps` | 5,000,000 | Training interaction budget |
| `seeds` | `(0, 1, 2)` | Minimum replication set for reported results |
| `seed` | `0` | Default seed for one run |
| `num_eval_eps` | `1` | Episodes per evaluation event |
| `gamma` | `0.8` | Task-level discount shared across algorithms |
| `eval_freq_env_steps` | 50,000 | Target periodic-evaluation cadence |
| `protocol_version` | `1.0` | Audit tag for produced reports |

Training scripts may expose command-line overrides, but their defaults should
read this dataclass rather than copy its current numeric values.

## Periodic evaluation

The framework-neutral `EvalCallback` can be driven by any trainer that knows
its aggregate environment-step count:

```python
from omnipiano.configs import BenchmarkProtocolConfig
from omnipiano.integrations.eval_callback import EvalCallback

protocol = BenchmarkProtocolConfig()

periodic_eval = EvalCallback(
    env_id=env_id,
    env=eval_env,
    run_dir=run_dir,
    eval_seed=100,
    protocol=protocol,
    policy=lambda: policy,
    save_best_fn=lambda path: policy.save(path),
)

# Call this after a training update with the aggregate environment-step count.
stats, evaluated = periodic_eval.on_step(env_steps=50_000)
```

Before the next evaluation boundary, `evaluated` is `False` and the returned
statistics are NaN placeholders. At a boundary, the callback:

1. evaluates `protocol.num_eval_eps` episodes;
2. returns mean reward, F1, and episode length;
3. appends the event to `evaluations.npz`;
4. saves a new best model when mean observed reward improves; and
5. advances the next boundary by `eval_freq_env_steps`.

Periodic evaluation calls `evaluate_policy(..., true_reward=False)`, matching
the reward signal optimized during training. Final benchmark evaluation should
normally keep the default `true_reward=True`.

`evaluations.npz` contains `timesteps`, `results`, and `ep_lengths`. If the
evaluation environment also writes episode CSV files, add the training-step
axis after training:

```python
from omnipiano.integrations.eval_callback import postprocess_eval_csv

postprocess_eval_csv(run_dir)
```

The helper repeats each evaluation timestep for every evaluation episode,
joins rows in rollout order, skips already-processed files, and leaves final
evaluation CSVs untouched when their row count does not match.

## Reproducible reporting

For paper results, run training independently for every seed in
`BenchmarkProtocolConfig.seeds`, retain each run's `eval_summary.json`, and
report mean and standard deviation across training seeds. Multiple episodes
inside one evaluation call estimate within-policy rollout variation; they do
not replace independent training replications.

```{warning}
Do not present several evaluation episodes from one trained checkpoint as
independent seeds. The benchmark replication unit is an independently trained
policy.
```

Continue with [Stable-Baselines3 Evaluation](sb3.md) for SB3's native periodic
callback and vector-environment step semantics.
