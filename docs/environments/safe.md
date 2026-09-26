# Safe Environments

Safe environments formulate piano playing as a constrained Markov decision
process (CMDP). Musical reward and physical safety cost are separate signals,
allowing constrained algorithms to enforce an episode budget without changing
the piano objective.

```{figure} ../_static/images/safetyRL.png
:alt: OmniPiano safety semantics and cost settings
:width: 100%
:align: center
:class: bold-italic-caption

Safe RL in OmniPiano combines four physical safety semantics with event,
excess, and fraction cost settings.
```

```{important}
`SafetyWrapper` never subtracts safety cost from reward and never terminates
an episode because of cost. The algorithm receives musical reward normally and
reads the separate cost from `info`.
```

## At a glance

| Aspect | Safe RL |
| --- | --- |
| Interface | Gymnasium `Env` via `omnipiano.make(env_id)` |
| Task selection | Choose a safety `Task` with a semantic, cost setting, and budget |
| Registration | `omnipiano.safety.suite.register_task()`; safety IDs are registered on demand |
| Evaluation | `mode="eval"` enables musical F1; safety cost remains in `info` |

## Register an environment

The current safety suite is factorized and registered on demand. Importing
`omnipiano` does not automatically register these tasks. Select a task from
a suite manifest and call `register_task()` in every process that constructs
the environment:

```python
import omnipiano
from omnipiano.safety.suite import MAIN, register_task

task = MAIN[0]
env_id = register_task(task)
env = omnipiano.make(env_id, seed=1)
obs, info = env.reset(seed=1)
env.close()
```

Re-registering the same task is safe and returns the same ID after validation.

### Register the complete suite

Use `register_all()` when a launcher needs to discover every safety task:

```python
from omnipiano.safety.suite import register_all

tasks_by_id = register_all()
for env_id, task in sorted(tasks_by_id.items()):
    print(env_id, task.budget)
```

`register_all()` combines all manifests, removes duplicate IDs, registers
each unique task, and returns an `{env_id: Task}` mapping.

```{tip}
Prefer `register_task()` for one experiment. Use `register_all()` only
for catalogue discovery or batch launchers.
```

## Supported settings

### Safety semantics

A `CostSpec` selects one physical semantic:

| Semantic | Measurement units | Default threshold/reference |
| --- | --- | --- |
| `joint_range` | THJ2, FFJ2, MFJ2, RFJ2, and LFJ2 positions on each selected hand | Outside the central 50% of the native joint range |
| `actuator_power` | Mechanical power `abs(force * velocity)` | Total reference `4 × hands` for event/excess; per-hand reference `4` for fraction |
| `injured_finger` | THJ1–THJ5 actuator power on one protected right hand | Per-actuator reference `1.0` |
| `hand_collision` | Normal contact force for every unordered hand pair | Pair-force reference `10.0` |

These are soft costs. “Injured finger” does not disable an actuator, and
“joint range” does not physically clamp the joint.

### Cost settings

For non-negative normalized excess values `e_i`, the setting computes:

| Setting | Per-step cost | Interpretation |
| --- | --- | --- |
| `event` | `1[any e_i > 0]` | Whether any violation exists at this step |
| `fraction` | `mean(1[e_i > 0])` | Fraction of monitored units in violation |
| `excess` | `mean(e_i)` | Mean normalized violation severity |

All four semantics support all three settings, producing a 4 × 3 catalogue.
The optional weight is applied after aggregation.

```{note}
For actuator power, event/excess measure total-system power, whereas fraction
measures the share of selected hands above a per-hand threshold. They share a
safety theme but not an identical measurement unit.
```

## Environment IDs

Safety IDs are derived from the complete task specification:

```text
OmniPiano-Safety-{Song}-{N}H-{semantic}-{setting}-{hash}-v1
```

The hash covers the number of hands, song, cost specification, episode budget,
and hand layout. Do not guess it manually; use `task.env_id` or the value
returned by `register_task()`.

```{warning}
Legacy IDs such as `OmniPiano-ClairDeLune-CollisionSafe-v0`,
`...-WristLimit-v0`, and `...-PowerConstrained-v0` do not identify
tasks in the new factorized safety suite. Use
`omnipiano.safety.suite.register_task`.
```

### Suite manifests

| Manifest | Contents |
| --- | --- |
| `MAIN` | Eight selected 2–5 hand tasks used by the main experiment |
| `HANDS` | Nested 2–5 hand actuator-power sensitivity tasks |
| `THRESHOLDS` | Five episode budgets for a fixed four-hand power task |
| `EXTENSIONS` | Four hand/song anchors across all 12 semantic/setting combinations |

Default episode budgets are proportional to nominal song length:
`0.05 × T` for event/fraction and `0.02 × T` for excess. The budget is
stored on the immutable safety `Task`, not in the core `SafetyConfig`.

## Define a custom task

The suite's `task()` helper applies the same validation and ID generation:

```python
import omnipiano
from omnipiano.safety.suite import register_task, task

spec = task(
    hands=3,
    song="PolonaiseOp40No1",
    semantic="hand_collision",
    setting="excess",
    budget=12.0,
)
env_id = register_task(spec)
env = omnipiano.make(env_id, mode="eval", seed=0)
obs, info = env.reset(seed=0)
env.close()
```

Supported songs are `ForElise`, `ClairDeLune`,
`PicturesGreatKiev`, and `PolonaiseOp40No1`; hand counts are 2–5.

Internally, `register_task()` registers explicit N-hand specs, OT fingering,
the safety protocol's zero energy reward penalty, and one
`SemanticConstraint` in the existing `SafetyConfig`. Older classes in
`omnipiano.safety.constraints` remain for legacy compatibility but are not
the recommended registration surface.

## Run an environment

Use `mode="eval"` whenever final musical F1 is required:

```python
import omnipiano
from omnipiano.safety.suite import MAIN, register_task

task = MAIN[0]
env_id = register_task(task)
env = omnipiano.make(env_id, mode="eval", seed=1)
obs, info = env.reset(seed=1)

terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    step_cost = info["step_safety/cost_total"]

print("musical F1:", info["episode_task/f1"])
print("episode cost:", info["episode_safety/cost_total"])
print("episode budget:", task.budget)
env.close()
```

## Evaluation and metrics

```{important}
The default `mode="train"` emits safety costs but does not emit
`episode_task/f1`, key precision/recall, or sustain F1. Request
`mode="eval"` before reading those musical metrics.
```

| Key pattern | Meaning |
| --- | --- |
| `step_safety/cost_<semantic>_<setting>` | Cost from the configured semantic/setting |
| `step_safety/<semantic>/unit_count` | Number of monitored units |
| `step_safety/<semantic>/raw_max` | Maximum raw measurement at this step |
| `step_safety/<semantic>/violating_fraction` | Fraction of units with positive excess |
| `step_safety/cost_total` | Sum of active constraint costs |
| `episode_safety/cost_total` | Cost accumulated across the episode |
| `episode_safety/violations` | Steps with at least one positive cost |

A complete result should report task ID, semantic/setting, threshold parameters,
episode budget, episode cost, violation count, and musical F1. A policy that
satisfies the budget by remaining inactive is a trivially safe failure.

## Runtime requirements

The OmniSafe stack uses an isolated Python 3.10 environment because its pinned
dependencies differ from the general benchmark. Follow
`omnipiano/safety/QUICKSTART.md` in the source repository before running
reference algorithms. After preparing PIG, verify the registered suite with:

```bash
python -m omnipiano.safety.smoke --steps 2 --out safety_smoke.json
```

See [Cross-Framework Evaluation](../evaluation/cross_framework.md) for common
final-report semantics.
