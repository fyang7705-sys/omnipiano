# Robust Environments

Robust environments extend a standard OmniPiano task with explicit
perturbations to the agent-environment interaction. The musical objective,
hand morphology, Gymnasium API, and evaluation metrics remain unchanged. This
lets a clean and a perturbed environment differ only in the registered
`RobustConfig`.

```{figure} ../_static/images/robustRL.png
:alt: OmniPiano robust reinforcement learning perturbation framework
:width: 100%
:align: center
:class: bold-italic-caption

Robust RL in OmniPiano perturbs actions, observations, rewards, and physical
environment parameters while preserving the underlying piano task.
```

Robustness settings belong to the environment ID during training. Evaluation
may scale those registered magnitudes without creating a second family of
IDs, as described later under **Evaluation strength**.

```{note}
Robust environments keep the standard `env.step(action)` API. Perturbation
configuration is resolved from the registered ID, so an RL library does not
need to pass a special action/configuration dictionary on every step.
```

## At a glance

| Aspect | Robust RL |
| --- | --- |
| Interface | Gymnasium `Env` via `omnipiano.make(env_id)` |
| Task selection | Use a registered perturbation ID and distribution |
| Registration | `omnipiano.register()` with `RobustConfig` |
| Evaluation | `mode="eval"` enables metrics; `eval_noise_scale` varies test strength |

## Supported settings

### Perturbation targets

OmniPiano supports three agent-facing signal channels and four physical
environment parameters.

| Target | Meaning | Injection point | Frequency |
| --- | --- | --- | --- |
| Action | Noise is added to the canonical action before it reaches the environment | Gymnasium `RobustWrapper` | Every control step |
| Observation | Noise is added to selected numeric observations before the observation dictionary is flattened | dm_env `DmEnvObsNoiseWrapper` | Every control step |
| Reward | Noise changes the reward observed by the policy while the true task reward remains available for evaluation | Gymnasium `RobustWrapper` | Every control step |
| Gravity | Additive change to vertical gravity | `OmniPianoTask` physics layer | Per episode or per step |
| Contact friction | Additive change to fingertip-key sliding friction | `OmniPianoTask` physics layer | Per episode or per step |
| Initial hand Y position | Independent lateral offset sampled for each hand | Task construction/reset | Per episode |
| Initial hand Z position | Independent height offset sampled for each hand | Task construction/reset | Per episode |

Observation noise is injected before `ConcatObservationWrapper`. The wrapper
can therefore operate on named observations and avoid categorical or counter
fields that should not receive continuous noise. Action, observation, and
environment random-number streams are derived independently from the master
environment seed.

By default, observation noise selects keys containing `joints_pos`,
`joints_vel`, `piano/state`, or `piano/sustain_state`. It deliberately skips
the score goal, step counter, previous action, and previous reward fields.

### Noise distributions

Every supported target can use one of three distributions:

| Distribution | Configuration | Interpretation |
| --- | --- | --- |
| Gaussian | `*_noise_std=sigma` | Zero-mean normal noise with standard deviation `sigma` |
| Uniform | `*_noise_uniform_low=lo`, `*_noise_uniform_high=hi` | Noise sampled directly from `[lo, hi]`; asymmetric intervals are valid |
| Shift | `*_noise_shift=value` | A deterministic additive offset |

`RobustConfig.noise_dist` sets the shared default distribution. Signal
channels may override it with `action_noise_dist`, `obs_noise_dist`, or
`reward_noise_dist`. Physical parameters provide equivalent overrides inside
`RobustEnvConfig`, such as `gravity_noise_dist`.

```{important}
Noise levels use each distribution's natural parameters: Gaussian standard
deviation, uniform lower/upper bounds, and a literal constant shift. Equal
numeric labels do not imply equal empirical variance across distributions.
```

Only magnitude fields belonging to the resolved distribution may be nonzero.
For example, a Gaussian action channel must use `action_noise_std`; combining
that with `action_noise_shift` in the same channel is rejected. Standard
deviations must be finite and non-negative, uniform bounds must be finite and
ordered, and a declared per-channel distribution must have a nonzero
magnitude.

When reward noise is active and `action_reward_observation=True`, OmniPiano
also replaces the previous-reward slot in the next observation with the
noised reward received by the policy. Reward-noise tasks currently require
`frame_stack=1`; registering reward noise with a larger frame stack raises
`NotImplementedError` rather than updating the wrong flattened frame.

## Register an environment

### Signal perturbations

The following registration adds Gaussian action noise with standard deviation
0.10:

```python
from omnipiano import register
from omnipiano.configs import RobustConfig

register(
    id="OmniPiano-ClairDeLune-A-Gauss-P10-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        noise_dist="gaussian",
        action_noise_std=0.10,
    ),
)
```

Uniform observation noise uses its natural lower and upper bounds:

```python
register(
    id="OmniPiano-ClairDeLune-O-Uniform-P20-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        noise_dist="uniform",
        obs_noise_uniform_low=-0.20,
        obs_noise_uniform_high=0.20,
    ),
)
```

A robust task may activate several channels at once. The next example uses a
different distribution and magnitude for each channel:

```python
register(
    id="OmniPiano-ClairDeLune-AOR-Mixed-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        action_noise_dist="shift",
        action_noise_shift=-0.05,
        obs_noise_dist="gaussian",
        obs_noise_std=0.20,
        reward_noise_dist="uniform",
        reward_noise_uniform_low=-0.30,
        reward_noise_uniform_high=0.30,
    ),
)
```

When using per-channel overrides, name every active channel explicitly. This
makes a mixed configuration readable without relying on the global fallback.

### Physical perturbations

Physical perturbations live in a nested `RobustEnvConfig`. This example
samples gravity once per episode, contact friction every control step, and the
initial hand pose once per episode:

```python
from omnipiano import register
from omnipiano.configs import RobustConfig, RobustEnvConfig

register(
    id="OmniPiano-ClairDeLune-GCFHP-Gauss-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        noise_dist="gaussian",
        environment_noise=RobustEnvConfig(
            gravity_noise_std=3.0,
            gravity_frequency="episode",
            contact_friction_noise_std=0.30,
            contact_friction_frequency="step",
            hand_position_y_noise_std=0.025,
            hand_position_z_noise_std=0.010,
        ),
    ),
)
```

Gravity is measured in `m/s^2`; hand-position offsets are measured in meters.
The registered values are additive noise magnitudes, not replacement physics
parameters. `gravity_frequency` and `contact_friction_frequency` accept only
`"episode"` or `"step"`. Initial hand position is always episode-level.

Signal and physical perturbations compose in one task:

```python
from omnipiano import register
from omnipiano.configs import (
    BenchmarkEnvConfig,
    RobustConfig,
    RobustEnvConfig,
)
from omnipiano.tasks.hand_spec import default_three_hand_specs

register(
    id="OmniPiano-ForElise-ThreeHand-ActionGravity-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    robust_config=RobustConfig(
        noise_dist="gaussian",
        action_noise_std=0.10,
        environment_noise=RobustEnvConfig(gravity_noise_std=3.0),
    ),
    env_config=BenchmarkEnvConfig(disable_fingering_reward=True),
    hand_specs=default_three_hand_specs(),
)
```

## Environment IDs

The canonical Clair de Lune signal sweep varies one factor at a time:

| Channel | ID code | Registered levels |
| --- | --- | --- |
| Action | `A` | 0.05, 0.10, 0.15 |
| Observation | `O` | 0.10, 0.20, 0.30 |
| Reward | `R` | 0.10, 0.30, 0.50 |

Each channel is paired with Gaussian, uniform, and shift perturbations. This
produces 27 single-channel environments plus the explicit clean baseline
`OmniPiano-ClairDeLune-Clean-v0`.

Physical-environment IDs use these channel codes:

| Code | Target | Example |
| --- | --- | --- |
| `G` | Gravity | `OmniPiano-ClairDeLune-G-Gauss-P300-v0` |
| `CF` | Contact friction | `OmniPiano-ClairDeLune-CF-Uniform-P30-v0` |
| `HP` | Initial hand Y/Z position | `OmniPiano-ClairDeLune-HP-Gauss-Y25-Z10-v0` |
| `GCFHP` | Compound physical perturbation | `OmniPiano-ClairDeLune-GCFHP-Gauss-G150-CF15-Y50-Z20-v0` |

`Step` in an environment ID denotes per-step resampling for the named physical
parameter. Without it, gravity and friction default to episode-level
resampling. Positive labels use `P`; directional negative shifts use `N`.
The integer is the natural magnitude multiplied by 100, except hand-position
`Y` and `Z` labels, which are expressed in millimeters.

## Register a sweep

Register related experiments programmatically so their naming and untouched
channels remain consistent:

```python
from omnipiano import register
from omnipiano.configs import RobustConfig


def signal_config(channel, distribution, level):
    fields = {"noise_dist": distribution}
    if distribution == "gaussian":
        fields[f"{channel}_noise_std"] = level
    elif distribution == "uniform":
        fields[f"{channel}_noise_uniform_low"] = -level
        fields[f"{channel}_noise_uniform_high"] = level
    elif distribution == "shift":
        fields[f"{channel}_noise_shift"] = level
    else:
        raise ValueError(f"unsupported distribution: {distribution}")
    return RobustConfig(**fields)


channels = {"action": "A", "obs": "O", "reward": "R"}
distributions = {
    "Gauss": "gaussian",
    "Uniform": "uniform",
    "Shift": "shift",
}

for channel, channel_code in channels.items():
    for dist_label, distribution in distributions.items():
        level = 0.10
        register(
            id=(
                "OmniPiano-ForElise-"
                f"{channel_code}-{dist_label}-P10-Custom-v0"
            ),
            base_env_name="RoboPianist-repertoire-150-ForElise-v0",
            robust_config=signal_config(channel, distribution, level),
        )
```

For official benchmark families, keep the loop and its helper in
`omnipiano/envs/__init__.py`. Registration happens when `omnipiano` is
imported, and duplicate IDs fail immediately.

## Run an environment

Training always uses the exact magnitudes stored in the registered
`RobustConfig`:

```python
from omnipiano import make

train_env = make(
    "OmniPiano-ClairDeLune-A-Gauss-P10-v0",
    mode="train",
    seed=0,
)
observation, info = train_env.reset(seed=0)
action = train_env.action_space.sample()
observation, reward, terminated, truncated, info = train_env.step(action)
train_env.close()
```

The policy still calls `step(action)`. Unlike Robust-Gymnasium, OmniPiano does
not require a dictionary containing an action and perturbation configuration
at every step. The registry resolves the robust settings before the rollout,
so standard Gymnasium-compatible trainers need no robust-specific adapter.

## Evaluation and metrics

### Evaluation strength

At evaluation time, `eval_noise_scale` multiplies every active signal and
physical magnitude:

```python
matched_env = make(
    "OmniPiano-ClairDeLune-A-Gauss-P10-v0",
    mode="eval",
    eval_noise_scale=1.0,
    seed=100,
)

clean_env = make(
    "OmniPiano-ClairDeLune-A-Gauss-P10-v0",
    mode="eval",
    eval_noise_scale=0.0,
    seed=100,
)

stress_env = make(
    "OmniPiano-ClairDeLune-A-Gauss-P10-v0",
    mode="eval",
    eval_noise_scale=2.0,
    seed=100,
)
```

| Scale | Meaning |
| ---: | --- |
| `0.0` | Clean evaluation of the same trained policy |
| `1.0` | Matched evaluation at the registered training strength; this is the default |
| `> 1.0` | Stress evaluation |

`eval_noise_scale` is a measurement parameter rather than part of task
identity, so a single trained policy can be measured over a full robustness
curve without registering many evaluation-only IDs. It is valid only with
`mode="eval"`, and it must be finite and non-negative.

```{tip}
Keep the policy checkpoint and evaluation seeds fixed across a noise-scale
sweep. Then the scale is the intended independent variable rather than one of
several simultaneous changes.
```

Reward noise is scaled in the same way as action and observation noise. The
unperturbed task reward remains available as `info["task/true_reward"]`, which
allows formal evaluation to report true musical return while preserving the
reward signal observed by the policy.

To record episode metrics, add `log_dir`:

```python
eval_env = make(
    "OmniPiano-ClairDeLune-O-Uniform-P20-v0",
    mode="eval",
    eval_noise_scale=1.5,
    seed=0,
    log_dir="runs/clair_de_lune/noise_1.5",
)
```

The CSV contains musical metrics, safety metrics when applicable, effective
noise scale, and accumulated perturbation statistics. Use the same evaluation
seed across scales when constructing a robustness curve.

## Registration rules

Robust environments follow the same registry contract as
[Standard Environments](standard.md):

- Put training-time perturbations in the registered `RobustConfig`.
- Register a new ID when a channel, distribution, magnitude, resampling
  frequency, base piece, morphology, or observation configuration changes.
- Do not pass `robust_config` or trajectory-affecting fields to `make()`.
- Use `eval_noise_scale` only to measure a registered task at different test
  strengths.
- Keep an explicit clean baseline in every experimental family.

These rules make the environment ID the reproducible description of training,
while still allowing clean, matched, and stress evaluation from one policy
checkpoint.
