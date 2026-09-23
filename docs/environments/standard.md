# Standard Environments

Standard environments provide the nominal OmniPiano task: a policy controls
one or more Shadow Hands to perform a MIDI score without robustness
perturbations. They use the same Gymnasium interface, musical objective, and
episode metrics as the robust and safe tracks, which makes a standard task the
natural control condition for later experiments.

An OmniPiano environment is not assembled from arbitrary arguments at every
call. Instead, a registered environment ID points to a complete `TaskSpec`.
The ID therefore records the piece, morphology, task variant, and all
trajectory-affecting environment settings.

```{note}
Treat the registered environment ID as the single source of truth for the
task. If a setting can change the policy's trajectory, encode it in a new
registration rather than passing it only from a training script.
```

## Supported settings

### Task and morphology

| Setting | Supported values | Configuration surface |
| --- | --- | --- |
| Musical piece | RoboPianist debug pieces or PIG repertoire pieces | `base_env_name` |
| Number of hands | One to five Shadow Hands | `hand_specs` |
| Hand side and pose | Left/right model, position, orientation, attachment yaw | `HandSpec` |
| Forearm control | Selectable forearm degrees of freedom and reduced/full action space | `HandSpec.forearm_dofs`, `HandSpec.reduced_action_space` |
| Workspace | Full-keyboard reach or a static keyboard partition | `HandSpec.key_range` or `HandSpec.y_range` |
| Logical grouping | Arbitrary labels for later multi-agent partitioning | `HandSpec.group` |
| Fingering objective | Annotation-based fingering or OT fingering | `BenchmarkEnvConfig.disable_fingering_reward` |
| Hand availability | Both hands active, left hand immobile, or right hand immobile | `TaskVariantConfig` |

OmniPiano provides `default_two_hand_specs()`,
`default_three_hand_specs()`, `default_four_hand_specs()`, and
`default_five_hand_specs()` for its canonical layouts. A `HandSpec` with no
range can reach the full keyboard. Setting `key_range=(lo, hi)` clamps the
hand's translational workspace to that inclusive key interval, where key 0 is
A0 and key 87 is C8.

The standard General-RL family uses the following layout ladder:

| Family | Layout |
| --- | --- |
| `OneHand-GeneralRL` | One hand selected from the canonical two-hand pair |
| `TwoHand-GeneralRL` | Canonical right/left pair |
| `ThreeHandPrototype-GeneralRL` to `FiveHandPrototype-GeneralRL` | Full-keyboard reach for every hand |
| `ThreeHand-StaticPartition-GeneralRL` to `FiveHand-StaticPartition-GeneralRL` | A fixed keyboard region for each hand |

```{tip}
Start with the provided `default_*_hand_specs()` layouts. Define custom
`HandSpec` objects only when the experiment specifically studies placement,
workspace partitioning, or agent grouping.
```

### Shared environment configuration

`BenchmarkEnvConfig` contains the settings shared by standard and robust
tasks. These values are registered as part of the experiment identity.

| Group | Fields | Default behavior |
| --- | --- | --- |
| Goal observation | `n_steps_lookahead` | Exposes 10 future score steps |
| Episode preparation | `trim_silence`, `stretch_factor`, `shift_factor` | Trims leading silence, keeps nominal tempo and pitch |
| Physics/control | `gravity_compensation`, `control_timestep` | Gravity compensation enabled at a 0.05 s control step |
| Action space | `reduced_action_space`, `clip` | Full hand action space, clipped to canonical `[-1, 1]` |
| Termination | `wrong_press_termination` | A wrong key does not terminate the episode |
| Rewards | `disable_fingering_reward`, `disable_forearm_reward` | Annotation fingering and forearm reward enabled |
| Contacts and visuals | `disable_hand_collisions`, `primitive_fingertip_collisions`, `disable_colorization`, `change_color_on_activation` | Detailed collision geometry and activation feedback |
| Temporal observation | `frame_stack`, `action_reward_observation` | One frame plus previous action and reward |
| Recording | `record_dir`, `record_every`, `record_resolution`, `camera_id` | Disabled unless requested at runtime |

`disable_fingering_reward=True` does not remove all fingering guidance. In the
OmniPiano task family it selects the OT-based assignment used by the
General-RL and multi-hand tasks instead of requiring annotated per-finger
targets.

## Environment IDs

Official task IDs follow this general form:

```text
OmniPiano-{PieceName}-{TaskVariant}-v0
```

For example:

```text
OmniPiano-ForElise-TwoHand-GeneralRL-v0
OmniPiano-WinterWind-FourHandPrototype-GeneralRL-v0
OmniPiano-WinterWind-FourHand-StaticPartition-GeneralRL-v0
```

The ID must change whenever a trajectory-affecting setting changes. In
particular, two tasks with different hand layouts, fingering objectives,
lookahead windows, frame stacks, or control timesteps should not share an ID.

## Use a registered environment

Importing `omnipiano` imports `omnipiano.envs`, which executes the official
`register(...)` declarations. A registered standard task can then be created
through the public factory:

```python
from omnipiano import make

env = make(
    "OmniPiano-WinterWind-FourHand-StaticPartition-GeneralRL-v0",
    seed=42,
)
observation, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

The observation is a flat `float32` box and the action is a canonical
`[-1, 1]` box. At the end of an evaluation episode, `info` includes musical
metrics such as `episode_task/f1`, `episode_task/key_precision`, and
`episode_task/key_recall`.

## Register a standard environment

The shortest registration needs only a unique ID and a RoboPianist base task:

```python
from omnipiano import register

register(
    id="OmniPiano-ForElise-MyStandardTask-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
)
```

With no optional configuration, the environment uses the canonical two-hand
layout and default `BenchmarkEnvConfig`, `TaskVariantConfig`, `RobustConfig`,
and `SafetyConfig`. The resulting task is clean because all robustness
magnitudes and all safety constraints are empty by default.

The following example registers a three-hand OT-fingering task with a custom
static partition and a two-frame observation:

```python
from omnipiano import register
from omnipiano.configs import BenchmarkEnvConfig
from omnipiano.tasks.hand_spec import HandSpec
from robopianist.models.hands import HandSide

register(
    id="OmniPiano-ForElise-ThreeHand-CustomPartition-v0",
    base_env_name="RoboPianist-repertoire-150-ForElise-v0",
    env_config=BenchmarkEnvConfig(
        disable_fingering_reward=True,
        frame_stack=2,
    ),
    hand_specs=(
        HandSpec(
            name="lh",
            side=HandSide.LEFT,
            position=(0.4, -0.30, 0.13),
            key_range=(0, 29),
            group="bass",
        ),
        HandSpec(
            name="rh_c",
            side=HandSide.RIGHT,
            position=(0.4, 0.00, 0.13),
            key_range=(30, 58),
            group="middle",
        ),
        HandSpec(
            name="rh",
            side=HandSide.RIGHT,
            position=(0.4, 0.30, 0.13),
            key_range=(59, 87),
            group="treble",
        ),
    ),
)
```

For an official benchmark task, place the declaration in
`omnipiano/envs/__init__.py` so importing the package registers it in every
process. For a local experiment, call `register()` once during application
startup before `make()`. Registering the same ID twice raises `ValueError`.

### Register task variants

OmniPiano-only structural variants use `TaskVariantConfig`. For example, a
right-hand-only task keeps the left hand in the model but makes it immobile:

```python
from omnipiano import register
from omnipiano.configs import TaskVariantConfig

register(
    id="OmniPiano-NocturneOp9No2-RightHandOnly-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-NocturneOp9No2-v0",
    task_config=TaskVariantConfig(left_hand_immobile=True),
)
```

### Register families without duplication

Task families should share a small helper so every ID receives the same base
configuration:

```python
from omnipiano import register
from omnipiano.configs import BenchmarkEnvConfig
from omnipiano.tasks.hand_spec import (
    default_two_hand_specs,
    default_three_hand_specs,
)


def register_hand_ladder(piece, base_env_name):
    config = BenchmarkEnvConfig(disable_fingering_reward=True)
    variants = {
        "TwoHand": default_two_hand_specs(),
        "ThreeHandPrototype": default_three_hand_specs(),
    }
    for name, hand_specs in variants.items():
        register(
            id=f"OmniPiano-{piece}-{name}-GeneralRL-v0",
            base_env_name=base_env_name,
            env_config=config,
            hand_specs=hand_specs,
        )
```

## Registry contract

`make()` deliberately accepts only runtime parameters that do not redefine the
experiment:

- `seed`
- `record_dir`
- `record_every`
- `record_resolution`
- `camera_id`

All other settings must come from the registered `TaskSpec`. For example,
`make(env_id, frame_stack=4)` is rejected; register a new ID with
`BenchmarkEnvConfig(frame_stack=4)` instead. This ensures that an environment
ID is sufficient to recover the task definition from an evaluation report.

```{warning}
Calling `register()` twice with the same ID raises `ValueError`. Put official
registrations in `omnipiano/envs/__init__.py`; keep local registrations in one
application startup path so subprocesses see the same task catalog.
```

Use `mode="train"` for training. Use `mode="eval"` to enable musical
precision/recall/F1 computation, and optionally pass `log_dir` to write one
CSV row per completed episode:

```python
eval_env = make(
    "OmniPiano-ForElise-TwoHand-GeneralRL-v0",
    mode="eval",
    seed=0,
    log_dir="runs/for_elise/eval",
)
```

Continue with [Robust Environments](robust.md) to add perturbations while
preserving the same task and registration contract.
