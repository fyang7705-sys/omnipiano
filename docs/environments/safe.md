# Safe Environments

Safe environments add explicit physical constraints to a standard OmniPiano
task. The task reward and safety cost are returned as separate signals, so the
same environment can be used with constrained RL algorithms such as CPO,
PPO-Lagrangian, and PPOLag, as well as with unconstrained baselines that log
the cost for analysis.

```{figure} ../_static/images/safetyRL.png
:alt: OmniPiano safety semantics and cost settings
:width: 100%
:align: center
:class: bold-italic-caption

Safe RL in OmniPiano covers joint, power, injury, and collision semantics with
event, excess, and fractional cost formulations.
```

```{important}
`SafetyWrapper` does **not** subtract the safety cost from the reward. The
environment returns the musical reward normally and writes costs to `info`.
The training algorithm decides how to enforce or trade off the constraint.
```

## Register a safe environment

Safety constraints are registered inside `SafetyConfig` and attached to a
normal `TaskSpec`. The shortest example uses binary hand-hand collision cost:

```python
from omnipiano import register
from omnipiano.configs import SafetyConfig
from omnipiano.safety.constraints import HandCollisionConstraint

register(
    id="OmniPiano-ClairDeLune-CollisionSafe-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    safety_config=SafetyConfig(
        constraints=[HandCollisionConstraint(penalty_coef=1.0)],
    ),
)
```

For official benchmark tasks, put the declaration in
`omnipiano/envs/__init__.py`. Local experiments may call `register()` once at
application startup, before `make()`. As with standard and robust tasks, the
environment ID should change whenever a trajectory-affecting configuration
changes.

## Available constraint types

| Constraint | Constructor idea | Cost interpretation | `info` key |
| --- | --- | --- | --- |
| `JointMagnitudeConstraint` | one action index and a maximum magnitude | excess normalized action magnitude | `step_safety/cost_joint_<index>_mag` |
| `MultiJointSharedMagnitudeConstraint` | joint group and per-joint ceiling | sum of each joint's excess magnitude | `step_safety/cost_group_<name>_mag` |
| `MultiJointSummedMagnitudeConstraint` | joint group and shared budget | excess over the group's summed magnitude budget | `step_safety/cost_group_<name>_sum_mag` |
| `HandCollisionConstraint` | penalty coefficient | fixed cost when any hand-hand contact exists | `step_safety/cost_hand_collision` |
| `HandCollisionForceConstraint` | penalty coefficient | continuous cost proportional to normal contact force | `step_safety/cost_hand_collision_force` |
| `TotalActuatorPowerConstraint` | penalty coefficient | dense total actuator power, `abs(force) * abs(velocity)` | `step_safety/cost_total_actuator_power` |
| `InjuredJointPowerConstraint` | hand, joint names, coefficient | power used by selected injury-sensitive actuators | `step_safety/cost_injured_<hand>_<joints>_power` |

Every configured constraint contributes its own per-step key. The
`SafetyWrapper` sums positive constraint costs into
`step_safety/cost_total` and counts steps with at least one violation in
`step_safety/violation_any`.

## Compose multiple constraints

Several constraints can be evaluated together. This example combines a
collision cost with a total-power cost:

```python
from omnipiano import register
from omnipiano.configs import SafetyConfig
from omnipiano.safety.constraints import (
    HandCollisionConstraint,
    TotalActuatorPowerConstraint,
)

register(
    id="OmniPiano-MapleLeafRag-CollisionPower-Custom-v0",
    base_env_name="RoboPianist-repertoire-150-MapleLeafRag-v0",
    safety_config=SafetyConfig(
        constraints=[
            HandCollisionConstraint(penalty_coef=1.0),
            TotalActuatorPowerConstraint(penalty_coef=0.1),
        ],
    ),
)
```

The total is a sum of the weighted costs returned by the individual
constraints. Choosing `penalty_coef` is therefore part of the task definition
and should be kept in the registration rather than changed at runtime.

## Run and inspect safety signals

Safe environments use the ordinary Gymnasium API:

```python
from omnipiano import make

env = make("OmniPiano-ClairDeLune-CollisionSafe-v0", seed=42)
obs, info = env.reset()

terminated = truncated = False
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
    musical_reward = reward
    safety_cost = info["step_safety/cost_total"]

if "episode_safety/cost_total" in info:
    print("episode safety cost:", info["episode_safety/cost_total"])
    print("unsafe steps:", info["episode_safety/violations"])
env.close()
```

At episode termination, `SafetyWrapper` adds:

| Episode key | Meaning |
| --- | --- |
| `episode_safety/cost_total` | Sum of all step safety costs |
| `episode_safety/violations` | Number of steps where at least one constraint had positive cost |

Musical metrics remain under `episode_task/*`, for example
`episode_task/f1`, `episode_task/key_precision`, and
`episode_task/key_recall`. This makes it possible to report both feasibility
and musical quality rather than hiding one inside the other.

## Built-in safe task families

The repository registers representative safe tasks including:

| Family | Example IDs | Main question |
| --- | --- | --- |
| Wrist or joint limits | `OmniPiano-ForElise-WristLimit-v0` | Can the policy play while limiting a selected joint magnitude? |
| Binary collision avoidance | `OmniPiano-ClairDeLune-CollisionSafe-v0` | Can the two hands avoid any contact? |
| Continuous collision force | `...-CollisionForce-v0` | Can contact severity be reduced smoothly rather than only counted as 0/1? |
| Total power budget | `...-PowerConstrained-v0` | Can the policy reduce actuator energy while preserving F1? |
| Injury-style local power | `...-WristInjury-v0`, `...-ThumbInjury-v0` | Can selected joints remain low-power during performance? |
| Grouped OT-fingering limits | `...-WristMiddleLimitOT-v0`, `...-WristThumbBudgetOT-v0` | Can fingering adapt under shared or summed joint budgets? |

The exact registered catalog is defined in `omnipiano/envs/__init__.py`; use
those IDs directly rather than reconstructing a safety configuration in a
training script.

## Safe RL integration notes

An algorithm that supports costs should read `step_safety/cost_total` at every
step and use `episode_safety/cost_total` for episode-level reporting. Do not
replace the environment reward with `reward - cost` unless you are explicitly
implementing an unconstrained ablation and label it as such.

For final comparison, evaluate safe policies with `mode="eval"` so musical
metrics are available, and report at least:

- musical F1 (and precision/recall);
- mean and maximum episode safety cost;
- number of violating steps; and
- the constraint family and penalty coefficients.

See [Cross-Framework Evaluation](../evaluation/cross_framework.md) for the
common evaluator and [Stable-Baselines3 Evaluation](../evaluation/sb3.md) for
training/evaluation environment separation.
