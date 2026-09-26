# Multi-Agent Environments

OmniPiano exposes cooperative multi-hand tasks through the PettingZoo
`ParallelEnv` API. Each environment decomposes a registered N-hand piano task
into decentralized agents, gives every agent its own observation
and action space, and keeps one shared musical objective.

```{figure} ../_static/images/maRL.png
:alt: OmniPiano multi-agent reinforcement learning settings
:width: 100%
:align: center
:class: bold-italic-caption

Multi-agent OmniPiano varies observability, coupling, heterogeneity, and
scalability through agent-specific observation and action territories.
```

```{note}
Single-agent and multi-agent environments have different APIs. Use
`omnipiano.make()` for a Gymnasium environment and
`omnipiano.make_parallel()` for a PettingZoo `ParallelEnv`.
```

## At a glance

| Aspect | Multi-Agent RL |
| --- | --- |
| Interface | PettingZoo `ParallelEnv` via `omnipiano.make_parallel(env_id)` |
| Task selection | Choose a registered Territorial ID or compile a custom SCHO task |
| Registration | `register_parallel()` for reusable IDs; custom tasks use `compile_task()` and `make_parallel_from_task()` |
| Evaluation | Shared reward per agent; terminal musical metrics in `infos["_global_"]` |

## Supported settings

### Agent layouts

The current Territorial series groups spatially adjacent hands into agents:

| Morphology | Setup | Agents and owned hands | Sustain owner |
| --- | --- | --- | --- |
| Three hand | `MainSolo` | `secondo=(lh, rh_c)`, `treble_soloist=(rh)` | `secondo` |
| Four hand | `Duet` | `secondo=(lh_b, rh_b)`, `primo=(lh_t, rh_t)` | `secondo` |
| Five hand | `Trio` | `left_secondo=(lh_b, rh_b)`, `center_soloist=(rh_c)`, `right_primo=(lh_t, rh_t)` | `left_secondo` |

Only the sustain owner receives the sustain dimension in its action space.
For the default 22-dimensional hand action layout, a two-hand sustain owner
has 45 actions, a two-hand non-owner has 44, and a one-hand agent has 22.

## Environment IDs

Multi-agent IDs follow this pattern:

```text
OmniPiano-{Piece}-{Morphology}-MA-{AgentSetup}-Territorial-v0
```

The built-in catalog currently includes:

| Morphology | Registered pieces |
| --- | --- |
| Three-hand MainSolo | `WinterWind`, `PicturesGreatKiev`, `PolonaiseOp40No1`, `PianoSonataNo281StMov` |
| Four-hand Duet | `WinterWind`, `PianoSonataNo301StMov`, `PicturesGreatKiev` |
| Five-hand Trio | `WinterWind` |

List the IDs registered in the running process with:

```python
from omnipiano.multiagent import list_parallel_envs

for env_id in list_parallel_envs():
    print(env_id)
```

## Run an environment

PettingZoo parallel environments consume one action per active agent and
return dictionaries keyed by agent name:

```python
import omnipiano

env = omnipiano.make_parallel(
    "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0",
    seed=42,
)

observations, infos = env.reset()

while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

Each `rewards[agent]` currently contains the same shared scalar musical
reward. Termination and truncation are synchronized across all agents. When
the episode finishes, `env.agents` is cleared according to the PettingZoo
convention.

```{tip}
Build the action dictionary from `env.agents`, not `env.possible_agents`.
`env.agents` is the authoritative set of agents that are active at the
current step.
```

## Observation and action structure

By default, each agent receives a dictionary observation containing:

| Field | Contents |
| --- | --- |
| `own_hands` | Joint observations for every hand controlled by the agent |
| `boundary_hands` | Joint observations for neighboring agents' hands nearest an inter-agent boundary |
| `piano_state` | Piano-key state sliced to the agent's reachable region |
| `sustain_state` | Global sustain-pedal state |
| `goal` | Score lookahead for reachable keys plus the global sustain target |
| `prev_action` | Agent-local previous action, when action/reward observation is enabled |
| `prev_reward` | Previous shared reward, when action/reward observation is enabled |

The `own_plus_boundary` design gives an agent its local state and the minimum
neighbor state needed for coordination near territorial boundaries. Each
agent's `info` also contains `agent_key_range`, which maps its local piano-key
slice back to global key indices.

Actions are normalized `float32` boxes in `[-1, 1]`. `make_parallel()`
reassembles the per-agent actions into the underlying global hand-action
vector in spatial order and appends sustain from the designated owner.

Set `flatten_obs=True` when a framework requires one flat `Box` observation
per agent:

```python
import omnipiano

env = omnipiano.make_parallel(
    "OmniPiano-WinterWind-FourHand-MA-Duet-Territorial-v0",
    seed=42,
    flatten_obs=True,
)
```

## Evaluation and metrics

At the terminal step, environment-wide musical metrics are added under the
special `infos["_global_"]` entry:

```python
global_metrics = infos.get("_global_", {})
print(global_metrics.get("episode_task/musical_f1"))
print(global_metrics.get("episode_task/musical_precision"))
print(global_metrics.get("episode_task/musical_recall"))
print(global_metrics.get("episode_task/sustain_f1"))
```

These are joint performance metrics for the whole ensemble, not independent
per-agent F1 scores.

## Register an environment

A multi-agent registration points to an existing single-agent StaticPartition
task and names one of the supported morphology assignments:

```python
from omnipiano.multiagent import register_parallel

register_parallel(
    id="OmniPiano-WinterWind-FourHand-MA-MyDuet-v0",
    sa_env_id="OmniPiano-WinterWind-FourHand-StaticPartition-v0",
    morphology="FourHand",
)
```

The referenced single-agent environment must already be registered and must
provide explicit N-hand `hand_specs` with a `key_range` for every hand.
OmniPiano combines each agent's hand ranges into one contiguous territory, so
hands owned by the same agent can coordinate within that shared region.

Official multi-agent declarations live in
`omnipiano/envs/multiagent_envs.py`. They are imported by
`omnipiano/envs/__init__.py`, so importing `omnipiano` registers the standard
multi-agent catalog as a side effect.

```{warning}
Use an ID containing `-MA-` with `make_parallel()`. Passing it to
`omnipiano.make()` raises an error that redirects to the PettingZoo factory;
passing a single-agent ID to `make_parallel()` produces the opposite redirect.
```

### Build a custom SCHO task

The updated multi-agent compiler can construct a task directly from a
versioned JSON task specification without mutating either registry. This is
the recommended surface for the SCHO studies of scalability, coupling,
heterogeneity, and observability.

Start from `omnipiano/multiagent/configs/marl_task_example.json`. Its `task`
block defines the song, hand count, agent-to-hand assignment, action ranges,
observation ranges, visible teammate hands, and sustain owner:

```python
import json
import omnipiano
from omnipiano.multiagent.compile import compile_task

with open("omnipiano/multiagent/configs/marl_task_example.json") as file:
    config = json.load(file)

env = omnipiano.make_parallel_from_task(
    compile_task(config["task"]),
    seed=0,
    flatten_obs=True,
)
observations, infos = env.reset(seed=0)
env.close()
```

`compile_task()` resolves the user-facing JSON `task` block into the task
snapshot expected by `make_parallel_from_task()`. This path supports custom agent counts and
explicitly overlapping action/observation territories while preserving a
fully serializable experiment definition.

## Runtime settings

`make_parallel()` accepts the normal non-trajectory runtime fields plus a
small multi-agent surface:

| Setting | Current behavior |
| --- | --- |
| `seed` | Initial environment seed |
| `record_dir`, `record_every`, `record_resolution`, `camera_id` | Optional sound/video recording settings |
| `obs_visibility` | Phase 1 supports only `"own_plus_boundary"` |
| `reward_mode` | Phase 1 supports only `"shared"` |
| `flatten_obs` | Preserve Dict observations or flatten each agent to a Box |
| `sustain_owner` | Override the morphology's default pedal-owning agent |
| `include_global_state` | Add centralized critic state; currently requires `flatten_obs=True` |
| `inter_agent_collision_penalty_coef` | Optional shared-reward penalty for inter-agent contact |
| `robust_config` | Optional observation-noise configuration; other robust channels are rejected |

Changing the sustain owner is useful for exceptional pieces or controlled
ablations, but the chosen name must be one of the agents in that morphology.

## Current scope and limitations

The current implementation is the first Territorial phase. Keep these
boundaries in mind:

- only `own_plus_boundary` observation visibility is implemented;
- only shared reward is implemented;
- registered Territorial tasks use static hand partitions; custom compiled
  tasks may define explicit agent action and observation ranges;
- single-agent safety constraints are rejected because the multi-agent chain
  does not yet expose `SafetyWrapper` costs;
- action-noise and reward-noise single-agent bases are rejected;
- physical environment perturbations are currently rejected;
- observation noise is supported, but does not yet have the single-agent
  `mode` / `eval_noise_scale` evaluation semantics; and
- `render()` itself returns `None`; use the recording arguments for videos.

These checks fail early instead of silently dropping costs or perturbations.

## Framework integration

The PettingZoo `ParallelEnv` can be adapted to MARL frameworks such as RLlib.
The current training entry point is configuration-driven:

```bash
python -m omnipiano.multiagent.train --list-algos
python -m omnipiano.multiagent.train --list-envs
python -m omnipiano.multiagent.train \
    omnipiano/multiagent/configs/marl_task_example.json --dry-run
```

Framework adapters commonly use `flatten_obs=True`; centralized-critic
methods may additionally request `include_global_state=True`.

```{important}
Do not reuse the single-agent cross-framework evaluator unchanged for a
multi-agent policy. It expects one observation and one action array, while a
PettingZoo parallel policy must produce an action dictionary for all active
agents.
```

See [Standard Environments](standard.md) for the underlying N-hand task and
registration contract.
