# Quick Start

This guide follows the installation and usage flow in the repository README.
It installs OmniPiano, prepares the Piano Fingering Dataset, and runs a first
Gymnasium environment.

## Create environment

OmniPiano targets Linux with Python 3.10 or newer. Windows users can follow the
same commands inside WSL2. Install the required native packages first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential fluidsynth libfluidsynth-dev portaudio19-dev ffmpeg libegl1 libgl1
```

Using Miniconda or Miniforge is recommended:

```bash
conda create -n pianist python=3.10 -y
conda activate pianist
```

## Clone and install OmniPiano

```bash
git clone https://anonymous.4open.science/r/omnipiano-A37E omnipiano
cd omnipiano
python -m pip install -e .
```

```{note}
The command above uses the anonymous review repository. It intentionally avoids
institution-identifying GitHub URLs during the submission period.
```

OmniPiano bundles the Shadow Hand model files and the default
`TimGM6mb.sf2` soundfont. You do not need to initialize a Git submodule or run
the original RoboPianist `install_deps.sh` script.

## Prepare the PIG dataset

Benchmark tasks use pieces from the
[Piano Fingering Dataset (PIG)](https://beam.kisarazu.ac.jp/research/PianoFingeringDataset/).
Download `PianoFingeringDataset_v1.2.zip` from the dataset website, extract it,
and preprocess the fingering files:

```bash
robopianist preprocess --dataset-dir /PATH/TO/PianoFingeringDataset_v1.2
robopianist --check-pig-exists
```

## Verify the installation

The registered Twinkle task provides a quick smoke test without the PIG dataset:

```bash
python -c "import omnipiano; env = omnipiano.make('OmniPiano-TwinkleTwinkleLittleStar-TwoHand-GeneralRL-v0'); print(env.action_space); env.close()"
```

After preprocessing PIG, verify a benchmark task:

```bash
python -c "import omnipiano; env = omnipiano.make('OmniPiano-WinterWind-FourHand-StaticPartition-GeneralRL-v0'); print(env.observation_space); env.close()"
```

## Run a first environment

OmniPiano single-agent tasks follow the standard Gymnasium API:

```python
import omnipiano

env = omnipiano.make(
    "OmniPiano-WinterWind-FourHand-StaticPartition-GeneralRL-v0",
    mode="eval",
    seed=42,
)
observation, info = env.reset(seed=42)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

print("musical F1:", info["episode_task/f1"])
print("precision:", info["episode_task/key_precision"])
print("recall:", info["episode_task/key_recall"])
env.close()
```

```{important}
Musical F1, precision, recall, and sustain metrics are emitted only when the
environment is created with `mode="eval"`. The default `mode="train"` omits
these terminal metrics to avoid evaluation overhead.
```

## Try a safe environment

The factorized safety suite is registered explicitly and process-locally. Its
IDs use the form `OmniPiano-Safety-...-v1`; legacy safety IDs such as
`OmniPiano-ClairDeLune-CollisionSafe-v0` are not part of the new suite.
The OmniSafe reference algorithms use a separate Python environment; follow
`omnipiano/safety/QUICKSTART.md` in the cloned repository before training them.

```python
import omnipiano
from omnipiano.safety.suite import MAIN, register_task

task = MAIN[0]  # 2-hand ForElise, joint_range / fraction
env_id = register_task(task)
env = omnipiano.make(env_id, mode="eval", seed=1)
observation, info = env.reset(seed=1)

terminated = truncated = False
while not (terminated or truncated):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

print("environment:", env_id)
print("musical F1:", info["episode_task/f1"])
print("safety cost:", info["episode_safety/cost_total"])
print("episode budget:", task.budget)
env.close()
```

Reward and safety cost remain separate: the wrapper does not subtract cost
from the musical reward. See [Safe Environments](../environments/safe.md) for
the full registration and cost semantics.

The random policy above is only an interface demonstration. Meaningful musical
performance requires training a policy with an RL algorithm.
