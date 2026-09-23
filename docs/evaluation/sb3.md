# Stable-Baselines3 Evaluation

Stable-Baselines3 policies already implement OmniPiano's required
`predict(observation, deterministic=True)` interface. Final evaluation can
therefore use the cross-framework evaluator directly. During training,
however, SB3's native `EvalCallback` is the preferred integration because it
is synchronized with SB3's callback lifecycle, vectorized step counter, model
saving, and `Monitor` statistics.

This page separates two evaluation products:

| Stage | Implementation | Primary purpose |
| --- | --- | --- |
| Periodic evaluation | `stable_baselines3.common.callbacks.EvalCallback` | Learning curves and best-checkpoint selection |
| Final evaluation | OmniPiano `evaluate_policy()` or an equivalent explicit loop | Framework-comparable final task metrics |

```{note}
Use SB3's callback for in-training scheduling and OmniPiano's common evaluator
for the final report. They complement each other and intentionally produce
different artifacts.
```

## Build separate training and evaluation environments

Do not evaluate on a training vector environment. Construct a single-worker
evaluation environment with `mode="eval"` so musical metrics and evaluation
CSV logging are active.

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from omnipiano import make


def env_creator(env_id, log_dir, mode):
    def create():
        return make(env_id, log_dir=log_dir, mode=mode)

    return create


env_id = "OmniPiano-ClairDeLune-Clean-v0"
log_dir = "runs/sb3/example"
n_envs = 16
seed = 0

train_env = make_vec_env(
    env_creator(env_id, log_dir, "train"),
    n_envs=n_envs,
    seed=seed,
    vec_env_cls=SubprocVecEnv,
)

eval_env = make_vec_env(
    env_creator(env_id, log_dir, "eval"),
    n_envs=1,
    seed=seed + 10_000,
)
```

`make_vec_env` wraps each environment with SB3's `Monitor`. This matters
because OmniPiano's `SafeRecordEpisodeStatistics` writes CSV rows but does not
populate SB3's `info["episode"]` record. The two wrappers coexist: `Monitor`
feeds SB3's callback statistics, while OmniPiano records musical, safety, and
robustness metrics.

For a robust evaluation environment, pass `eval_noise_scale` through the
factory used for `eval_env`. Training must continue to use the registered
training strength.

## Configure periodic evaluation

SB3 calls a callback once per vector-environment step, while OmniPiano's
protocol expresses cadence in aggregate environment interactions. Convert the
frequency before constructing `EvalCallback`:

```python
from stable_baselines3.common.callbacks import EvalCallback

eval_freq_env_steps = 50_000
eval_freq_vec_steps = max(eval_freq_env_steps // n_envs, 1)

periodic_eval = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=eval_freq_vec_steps,
    n_eval_episodes=1,
    deterministic=True,
    render=False,
)
```

Pass the callback to training:

```python
model.learn(
    total_timesteps=5_000_000,
    callback=periodic_eval,
)
```

With vectorized training, SB3 increments `num_timesteps` by `n_envs` on each
callback step. Dividing the requested aggregate cadence by `n_envs` prevents a
16-worker run from evaluating 16 times less often than a single-worker run.
Some on-policy algorithms can only expose useful checkpoints at rollout
boundaries, so the realized curve spacing may be slightly larger than the
nominal interval.

```{warning}
SB3's `eval_freq` counts callback calls, not aggregate environment
interactions. Always divide the protocol cadence by `n_envs` for vectorized
training.
```

Use `BenchmarkProtocolConfig` rather than hardcoding the defaults:

```python
from omnipiano.configs import BenchmarkProtocolConfig

protocol = BenchmarkProtocolConfig()
eval_freq_vec_steps = max(protocol.eval_freq_env_steps // n_envs, 1)

periodic_eval = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=eval_freq_vec_steps,
    n_eval_episodes=protocol.num_eval_eps,
    deterministic=True,
    render=False,
)
```

The same `num_eval_eps` value must reach both periodic and final evaluation;
otherwise the learning curve and final report use different protocols.

## Periodic artifacts

SB3's callback writes:

| Artifact | Contents |
| --- | --- |
| `best_model.zip` | Checkpoint with the highest periodic mean reward |
| `evaluations.npz` | `timesteps`, per-episode `results`, and `ep_lengths` |
| TensorBoard evaluation scalars | Mean reward, mean episode length, and optional success rate |

Because the evaluation environment was created with `mode="eval"` and a
`log_dir`, OmniPiano also writes `eval_episode_metrics_<env-id>.csv`. Its
columns include:

- received episode return and true episode return;
- episode length;
- safety cost and violations;
- key and sustain precision/recall/F1;
- reward decomposition;
- effective robust evaluation scale; and
- accumulated action, observation, and reward perturbation statistics.

The CSV's `env_step_count` is local to the evaluation environment. It is not
the training timestep. SB3's `evaluations.npz["timesteps"]` is authoritative
for the horizontal axis of a learning curve.

```{tip}
Run `postprocess_eval_csv(log_dir)` once training finishes. It joins the
authoritative SB3 timestep into the richer OmniPiano episode CSV without
changing the original evaluation arrays.
```

After training, join that axis into the CSV:

```python
from omnipiano.integrations.eval_callback import postprocess_eval_csv

postprocess_eval_csv(log_dir)
```

The helper supports more than one evaluation episode by repeating each SB3
event timestep for all episodes in that event.

## Best model semantics

SB3's native `EvalCallback` ranks checkpoints by the scalar reward returned by
the environment. On a reward-robust task, that is the perturbed reward observed
by the policy, not `info["task/true_reward"]`.

This is intentional for an in-training selection criterion: the checkpoint is
chosen using the objective experienced during training. It is distinct from
the final benchmark report, which should accumulate true task reward and
musical metrics. Record both definitions explicitly when reward perturbations
are part of an experiment.

```{important}
For reward-noise experiments, label received return and true return separately.
Using the same word "return" for both can reverse the interpretation of a
robustness result.
```

## Run final benchmark evaluation

After training, use a fresh, non-vectorized OmniPiano environment and the
cross-framework evaluator:

```python
from pathlib import Path

from omnipiano import make
from omnipiano.integrations.eval_callback import (
    evaluate_policy,
    write_eval_summary,
)

run_dir = Path(log_dir)
eval_seed = seed + n_envs + 1

final_env = make(
    env_id,
    mode="eval",
    seed=eval_seed,
    log_dir=str(run_dir),
    record_dir=str(run_dir / "videos"),
)

try:
    result = evaluate_policy(
        model,
        env_id,
        final_env,
        eval_seed=eval_seed,
        num_episodes=protocol.num_eval_eps,
    )
finally:
    final_env.close()

result.update(
    {
        "algorithm": model.__class__.__name__,
        "seed": seed,
        "total_env_steps": protocol.total_env_steps,
        "protocol_version": protocol.protocol_version,
    }
)
write_eval_summary(result, run_dir / "eval_summary.json")
```

The default `true_reward=True` makes `episode_return` comparable across clean
and reward-perturbed evaluations. The terminal metrics remain the primary
musical-quality measurements.

For a robustness curve, reconstruct the final environment at every desired
`eval_noise_scale` while keeping the checkpoint and `eval_seed` fixed:

```python
for scale in (0.0, 0.5, 1.0, 1.5, 2.0):
    env = make(
        env_id,
        mode="eval",
        eval_noise_scale=scale,
        seed=eval_seed,
        log_dir=str(run_dir / f"noise_{scale:g}"),
    )
    try:
        result = evaluate_policy(
            model,
            env_id,
            env,
            eval_seed,
            protocol.num_eval_eps,
        )
    finally:
        env.close()
```

## Recommended run contents

A complete SB3 run directory should retain:

```text
run_dir/
|-- best_model.zip
|-- final_model.zip
|-- evaluations.npz
|-- eval_episode_metrics_<env-id>.csv
|-- eval_summary.json
|-- progress.csv
|-- tensorboard/
`-- videos/
```

Add effective algorithm hyperparameters to `eval_summary.json`, including
network architecture, batch size, learning rate, replay-buffer settings,
gradient steps, and any command-line overrides. The environment ID records the
task configuration; it does not describe algorithm configuration.

## Common mistakes

- Passing `eval_freq_env_steps` directly to SB3 without dividing by `n_envs`.
- Reusing the training environment for evaluation.
- Omitting `mode="eval"`, which disables musical F1 computation.
- Treating evaluation-local `env_step_count` as the training timestep.
- Hardcoding one periodic episode while exposing a different
  `num_eval_eps` for final evaluation.
- Ranking reward-robust checkpoints by noisy reward and later reporting that
  value as true musical return.
- Reporting repeated evaluation episodes as independent training seeds.

See [Cross-Framework Evaluation](cross_framework.md) for the common result
schema and replication protocol shared with non-SB3 algorithms.
