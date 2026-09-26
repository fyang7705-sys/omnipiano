# Overview

OmniPiano is a unified benchmark for learning dexterous multi-hand piano
playing. Built on RoboPianist and the MuJoCo physics engine, it turns piano
performance into a shared testbed for standard, robust, safe, and cooperative
multi-agent reinforcement learning.

## 🧠Why piano playing?

Dexterous piano playing provides a suitable testbed by combining precise
control with measurable task outcomes:

- **Precise control.** The task requires coordinated finger movements to press,
  hold, and release keys at the times specified by a MIDI score.
- **Measurable task outcomes.** These score-defined targets allow execution
  accuracy to be measured by comparing the intended and actual key activations.
- **Systematic difficulty variation.** Control difficulty can be varied through
  musical complexity, the number and workspaces of the hands, and fingering
  freedom.

## 🔥Benchmark tracks

All four tracks share the same piano, MIDI objective, and evaluation basis, and
each varies one aspect of the task — so a performance difference can be
attributed to that single design choice.

- **Standard RL.** Scales repertoire and hand morphology: up to 150 pieces and one
  to five Shadow Hands, with fully mobile hands or static partitions that clamp
  each hand to a register. The reward combines key activation, fingertip-to-key
  matching, and actuation regularization, under annotation-based or
  optimal-transport (OT) fingering.
- **Robust RL.** Perturbs the agent–environment interaction at six targets —
  action, observation, reward, gravity, fingertip–key friction, and initial
  hand pose — each under Gaussian, bounded-uniform, or constant-shift noise.
  Channels compose into compound stress tests.
- **Safe RL.** Keeps musical reward and physical safety cost as separate signals,
  covering joint magnitude, inter-hand collision, actuator power, and
  joint-injury risk. Feasibility is judged before musical quality, and a policy
  that stays inactive to keep its cost low counts as a trivial-safe failure.
- **Multi-agent RL.** Distributes a fixed set of hands across decentralized agents
  that share one musical objective, exposed through a PettingZoo
  `ParallelEnv`. Cooperation difficulty is organized along scalability,
  coupling, heterogeneity, and observability (**SCHO**).

## 🔥Benchmark Features

- **Unified task family**: Standard, robust, safe, and multi-agent RL share one
  piano, one MIDI objective, and one evaluation basis, each varying a single
  aspect of the same task.
- **Scalable morphology**: One to five Shadow Hands — up to 111 continuous
  action dimensions — with optional static partitions that clamp each hand to a
  keyboard register.
- **Comprehensive task coverage**: Up to 150 pieces across eight hand settings,
  with at least 912 task settings spanning all four tracks.
- **High compatibility**: Gymnasium and PettingZoo APIs with vectorized
  environment support, usable from Stable-Baselines3, RLlib, CleanRL, OmniSafe,
  and TorchRL.

## 🔍 Comparison with Related RL Benchmarks

OmniPiano extends existing reinforcement-learning benchmarks by combining scalable dexterous control with robust RL, safe RL, multi-agent RL, and LLM-agent evaluation under a unified task setting.

| Feature                             | RoboPianist | Robust-Gymnasium | Safety-Gymnasium | **OmniPiano** |
| ----------------------------------- | :---------: | :--------------: | :--------------: | :-----------------: |
| 🎛️**Action dimension**      |     45     |      1–30      |      2–17      |  **23–11**  |
| 🎹**Dexterous piano playing** |     ✅     |        ❌        |        ❌        |    **✅**    |
| 🛡️**Robust RL track**       |     ❌     |        ✅        |        ❌        |    **✅**    |
| ⚠️**Safe RL track**         |     ❌     |        ✅        |        ✅        |    **✅**    |
| 🤝**Multi-Agent RL track**    |     ❌     |        ✅        |        ✅        |    **✅**    |
| 🔗**Unified task setting**    |     ❌     |        ❌        |        ❌        |    **✅**    |
| 🤖**LLM agent evaluation**    |     ❌     |        ❌        |        ❌        |    **✅**    |

> **OmniPiano scales from one to five Shadow Hands** , expanding the continuous action space from  **23 to 111 dimensions** .
> Episodes run at a  **20 Hz control frequency** , with horizons ranging from **240 to 3710 control steps** across the 150-piece repertoire.

### Supported baseline algorithms

Baseline algorithms supported by OmniPiano across its four evaluation tracks:

| Track | Baselines |
| --- | --- |
| Standard RL | *A2C*, ARS, <u>CrossQ</u>, <u>DDPG</u>, <u>DroQ</u>, *PPO*, <u>SAC</u>, <u>TD3</u>, <u>TQC</u>, *TRPO* |
| Robust RL | <u>A2P-SAC</u>, *EPPO*, <u>OMPO</u>, <u>SCPO</u> |
| Safe RL | *PPO-Lag*, *TRPO-Lag*, *RCPO*, *PDO*, *FOCOPS*, *CPO*, *PCPO*, *OnCRPO*, *IPO*, *P3O*, *CUP*, <u>DDPG-Lag</u>, <u>TD3-Lag</u>, <u>SAC-Lag</u> |
| Multi-Agent RL | *IPPO*, *MAPPO*, *HAPPO*, *MAT*, *A2PO*, <u>HATD3</u>, <u>FACMAC</u>, <u>MACSAC</u> |

*Italic* denotes on-policy methods; <u>underlined</u> denotes off-policy
methods. ARS is a derivative-free policy-search method.

## 🎬 Demonstrations

Swipe horizontally, use a trackpad, or focus a gallery and scroll with the
keyboard. Select a video to play it; each player has its own timeline and
volume controls.

### Standard RL

```{raw} html
<div class="demo-gallery" role="region" aria-label="Standard RL videos" tabindex="0">
  <figure class="demo-card">
    <video controls preload="none" poster="../featured-furelise-2hand.jpg" aria-label="Featured two-hand Für Elise performance"><source src="../featured-furelise-2hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Featured: Für Elise</strong><span>Two-hand piano performance</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../std-sac-furelise-1hand.jpg" aria-label="One-hand SAC playing Für Elise"><source src="../std-sac-furelise-1hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Für Elise · One hand</strong><span>Standard RL with SAC</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../std-sac-furelise-2hand.jpg" aria-label="Two-hand SAC playing Für Elise"><source src="../std-sac-furelise-2hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Für Elise · Two hands</strong><span>Standard RL with SAC</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../std-sac-greatkiev-3hand.jpg" aria-label="Three-hand SAC playing Pictures at an Exhibition Great Kiev"><source src="../std-sac-greatkiev-3hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Great Kiev · Three hands</strong><span>Standard RL with SAC</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../std-sac-winterwind-4hand.jpg" aria-label="Four-hand SAC playing Winter Wind"><source src="../std-sac-winterwind-4hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Winter Wind · Four hands</strong><span>Standard RL with SAC</span></figcaption>
  </figure>
</div>
```

### Robust RL

```{raw} html
<div class="demo-gallery" role="region" aria-label="Robust RL videos" tabindex="0">
  <figure class="demo-card">
    <video controls preload="none" poster="../rob-clairdelune-2hand-action.jpg" aria-label="Two-hand Clair de Lune under action noise"><source src="../rob-clairdelune-2hand-action.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Clair de Lune · Two hands</strong><span>Action noise</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../rob-furelise-3hand-obs-action.jpg" aria-label="Three-hand Für Elise under observation and action noise"><source src="../rob-furelise-3hand-obs-action.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Für Elise · Three hands</strong><span>Observation and action noise</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../rob-furelise-5hand-obs.jpg" aria-label="Five-hand Für Elise under observation noise"><source src="../rob-furelise-5hand-obs.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Für Elise · Five hands</strong><span>Observation noise</span></figcaption>
  </figure>
</div>
```

### Multi-Agent RL

```{raw} html
<div class="demo-gallery" role="region" aria-label="Multi-Agent RL videos" tabindex="0">
  <figure class="demo-card">
    <video controls preload="none" poster="../marl-two-agents-four-hands.jpg" aria-label="Two agents controlling four hands"><source src="../marl-two-agents-four-hands.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Two agents · Four hands</strong><span>Base cooperative setting</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../marl-heterogeneity.jpg" aria-label="Heterogeneous hand assignment with one hand versus three hands"><source src="../marl-heterogeneity.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>One hand vs. three hands</strong><span>Heterogeneity setting</span></figcaption>
  </figure>
  <figure class="demo-card">
    <video controls preload="none" poster="../marl-four-agents-one-hand.jpg" aria-label="Four agents controlling one hand each"><source src="../marl-four-agents-one-hand.mp4" type="video/mp4">Your browser does not support HTML5 video.</video>
    <figcaption><strong>Four agents · One hand each</strong><span>Scalability setting</span></figcaption>
  </figure>
</div>
```

Continue with the [Quick Start](quick_start.md) to install OmniPiano and run a
first environment.
