# SO-101 Arm Environment for PufferLib — End‑to‑End Build Plan

This document captures an implementation plan, coding patterns, and “line‑by‑line” guidance to build a SO‑101 (SO‑ARM100 family) robot arm environment for PufferLib, in two variants:

- A fast, kinematics‑only DH environment (no physics).
- A PyBullet physics environment that loads the official URDF from TheRobotStudio/SO-ARM100.

It also summarizes PufferLib’s environment architecture and coding style so we can align the PR cleanly.

---

## Goals

- Provide an RL environment that can: pick/place a small object using a SO‑101/SO‑100 arm; expose Gymnasium API; integrate with `pufferlib.vector` and `pufferlib.pufferl.PuffeRL`.
- Two backends:
  - DH kinematics: fast, useful for validating FK/IK and reward shaping.
  - PyBullet physics: cross‑platform on macOS with GUI for demo; headless for training.
- Make DH params configurable; accept URDF path; provide example training scripts and minimal tests.

---

## Sources and Geometry

- TheRobotStudio/SO-ARM100 (official):
  - CAD: STEP/STL available under `STEP/` and `STL/`.
  - URDF: under `Simulation/` (preferred, ready for sim) and `URDF/SO_5DOF_ARM100_8j_URDF.SLDASM/` (with meshes).
- SO‑100 DH Frames shared (standard Craig convention). We will:
  - Use the provided DH for SO‑100/101 if available; else extract from URDF or use Pinocchio for FK/Jacobian (skipping DH derivation).

---

## PufferLib Architecture and Style (observed)

- Environments live under `pufferlib/environments/<name>` with:
  - `environment.py` or module implementing a Gymnasium env or an `env_creator` factory.
  - `__init__.py` re‑exporting environment APIs.
- For Gymnasium envs that should work with the pufferl trainer, wrap with `pufferlib.emulation.GymnasiumPufferEnv`.
- Example pattern (see `pufferlib/environments/mujoco/environment.py`):
  - A `single_env_creator(...)` that builds/wraps a Gym env, including NormalizeObservation/Reward, `RecordVideo` for idx=0, etc.
  - An `env_creator(env_name=..., gamma=...)` returning `functools.partial(single_env_creator, **defaults)`.
- Minimal docstrings; clear, short function names; parameters are explicit with sensible defaults. Hardcoded paths avoided; pass URDF or config paths as args.
- Examples in `examples/` show how to create policies and trainers (e.g., `examples/pufferl.py`, `examples/vectorization.py`).

---

## Repo Layout to Add (proposed)

```
pufferlib/
  environments/
    so_arm_kin/
      __init__.py
      env.py                 # Kinematics‑only DH env
    so_arm_bullet/
      __init__.py
      environment.py         # PyBullet physics env + env_creator
examples/
  so_arm_kin_train.py        # PPO-style training with PuffeRL
  so_arm_bullet_train.py     # Same for physics variant
tests/
  environments/
    test_so_arm_kin.py       # (Optional) FK/Jacobian/IK checks
    test_so_arm_bullet_smoke.py (Optional) URDF load + reset/step
```

---

## Kinematics‑Only DH Environment (so_arm_kin)

### Observation/Action/Reward (task)

- Observations (float32): `[q(DoF), ee_pos(3), obj_pos(3), goal_pos(3), grasped(1)]`.
- Actions (float32): `Δq(DoF)` in radians per step plus a gripper scalar in `[-1, 1]` (`>0` close, `<0` open).
- Reward: `-||obj − goal|| + 1.0 * [reached] - 0.001 * ||Δq||`; reach if distance ≤ threshold; success or max steps terminates.

### Kinematics Math

- Standard DH transform:
  - `A_i(θ_i; a_i, α_i, d_i)` per link; FK is product of `A_i`.
  - End‑effector position `p = T[0:3, 3]`.
- Jacobian of position `J ∈ R^{3×n}` via finite differences initially (analytic optional later).
- Damped least squares IK step:
  - `Δθ = Jᵀ (J Jᵀ + λ² I)⁻¹ (p_goal − p)`; clamp per step; enforce joint limits.

### Env skeleton (line‑by‑line commentary)

```python
# env.py
import numpy as np
import gymnasium
import pufferlib

# 1) DH container and FK/Jacobian helpers
class DHParams:
    def __init__(self, a, alpha, d):
        self.a = np.asarray(a, float)
        self.alpha = np.asarray(alpha, float)
        self.d = np.asarray(d, float)
        assert self.a.shape == self.alpha.shape == self.d.shape

def _A(a, alpha, d, theta):
    # Build 4x4 DH matrix for given params/θ
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d ],
        [0,       0,      0,    1 ],
    ], float)

def fk_pos(q, dh: DHParams):
    # Multiply A_i; return end‑effector position and transforms
    T = np.eye(4)
    Ts = [T.copy()]
    for i in range(len(q)):
        T = T @ _A(dh.a[i], dh.alpha[i], dh.d[i], q[i])
        Ts.append(T.copy())
    return T[:3, 3].copy(), Ts

def jacobian_pos(q, dh: DHParams, eps=1e-5):
    # Finite‑difference Jacobian of position wrt θ
    J = np.zeros((3, len(q)))
    p0, _ = fk_pos(q, dh)
    for i in range(len(q)):
        dq = q.copy(); dq[i] += eps
        p1, _ = fk_pos(dq, dh)
        J[:, i] = (p1 - p0) / eps
    return J

def dls_step(q, p_goal, dh: DHParams, lam=1e-2, step_limit=0.05):
    # One DLS IK step for position only
    p, _ = fk_pos(q, dh)
    e = (p_goal - p).reshape(3, 1)
    J = jacobian_pos(q, dh)
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(3), e)
    dq = dq.squeeze()
    # Step clamp (avoid overshoot)
    norm = np.linalg.norm(dq, ord=np.inf)
    if norm > step_limit:
        dq *= step_limit / norm
    return dq

# 2) PufferEnv implementation
class SOArmKinEnv(pufferlib.PufferEnv):
    def __init__(self, buf=None,
                 dof=6,
                 dh=None,                 # dict with arrays a, alpha, d
                 joint_min=None, joint_max=None,
                 ws_min=(-0.6,-0.6,0.0), ws_max=(0.6,0.6,0.6),
                 max_steps=200,
                 grasp_radius=0.03,
                 ):
        # Configure DH and limits
        if dh is None:
            # Placeholder; in production load SO‑101/100 params
            dh = dict(a=[0, -0.425, -0.392, 0, 0, 0],
                      alpha=[np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0],
                      d=[0.089, 0, 0, 0.109, 0.095, 0.082])
        self.dh = DHParams(dh['a'], dh['alpha'], dh['d'])
        self.dof = dof
        lim = np.deg2rad(360*np.ones(dof)) if joint_min is None else None
        self.joint_min = (-lim if joint_min is None else np.asarray(joint_min, float))
        self.joint_max = ( lim if joint_max is None else np.asarray(joint_max, float))
        self.ws_min = np.asarray(ws_min, float)
        self.ws_max = np.asarray(ws_max, float)
        self.max_steps = int(max_steps)
        self.grasp_radius = float(grasp_radius)

        # State buffers
        self.q = np.zeros(dof, float)
        self.ee = np.zeros(3, float)
        self.obj = np.zeros(3, float)
        self.goal = np.zeros(3, float)
        self.grasped = False
        self.t = 0

        # Observation/action spaces
        obs_low  = np.concatenate([self.joint_min, self.ws_min, self.ws_min, self.ws_min, [0]])
        obs_high = np.concatenate([self.joint_max, self.ws_max, self.ws_max, self.ws_max, [1]])
        self.single_observation_space = gymnasium.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([-0.05]*dof + [-1], np.float32),
            high=np.array([ 0.05]*dof + [ 1], np.float32),
            dtype=np.float32,
        )
        self.num_agents = 1
        super().__init__(buf)

    def _obs(self):
        return np.concatenate([self.q, self.ee, self.obj, self.goal, [1.0 if self.grasped else 0.0]]).astype(np.float32)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.q = rng.uniform(-0.1, 0.1, size=self.dof)
        self.ee, _ = fk_pos(self.q, self.dh)
        self.obj = rng.uniform(self.ws_min, self.ws_max); self.obj[2] = max(self.obj[2], 0.05)
        self.goal = rng.uniform(self.ws_min, self.ws_max); self.goal[2] = max(self.goal[2], 0.05)
        self.grasped = False; self.t = 0
        self.observations[0] = self._obs()
        self.rewards[0] = 0; self.terminals[0] = False; self.truncations[0] = False
        return self.observations, [dict()]

    def step(self, actions):
        a = np.asarray(actions)[0]
        dq, grip = a[:self.dof], a[self.dof]
        self.q = np.clip(self.q + dq, self.joint_min, self.joint_max)
        self.ee, _ = fk_pos(self.q, self.dh)
        # Grasp toggle
        if grip > 0 and not self.grasped and np.linalg.norm(self.ee - self.obj) <= self.grasp_radius:
            self.grasped = True
        if grip < 0 and self.grasped:
            self.grasped = False
        if self.grasped:
            self.obj = self.ee.copy()
        # Reward/termination
        dist = np.linalg.norm(self.obj - self.goal)
        reached = dist <= 0.03
        r = -dist + (1.0 if reached else 0.0) - 0.001*np.linalg.norm(dq)
        self.t += 1
        done = bool(reached); trunc = bool(self.t >= self.max_steps)
        # Write buffers
        self.rewards[0] = r
        self.terminals[0] = done
        self.truncations[0] = trunc
        self.observations[0] = self._obs()
        info = dict(dist_to_goal=float(dist), reached=bool(reached), grasped=bool(self.grasped))
        return self.observations, self.rewards, self.terminals, self.truncations, [info]
```

### Example Trainer (so_arm_kin_train.py)

```
import torch, pufferlib.vector
from pufferlib import pufferl
from pufferlib.environments.so_arm_kin.env import SOArmKinEnv

class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(),
        )
        self.action_head = torch.nn.Linear(128, act_dim)
        self.value_head = torch.nn.Linear(128, 1)
    def forward(self, obs, state=None):
        x = self.net(obs); return self.action_head(x), self.value_head(x)

if __name__ == "__main__":
    env_creator = SOArmKinEnv
    vecenv = pufferlib.vector.make(env_creator, num_envs=8, num_workers=8, batch_size=8,
                                   backend=pufferlib.vector.Multiprocessing)
    policy = Policy(vecenv.driver_env).to('cuda' if torch.cuda.is_available() else 'cpu')
    args = pufferl.load_config('default'); args['train']['env'] = 'so_arm_kin'
    args['train']['batch_size'] = 4096; args['train']['bptt_horizon'] = 256
    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)
    for _ in range(5): trainer.evaluate(); trainer.train()
    trainer.print_dashboard(); trainer.close()
```

---

## PyBullet Physics Environment (so_arm_bullet)

### Responsibilities

- Load SO‑101/SO‑100 URDF (param: `urdf_path`), configure joint indices, gravity.
- Spawn a small cube (object) and a goal marker.
- Control joints with position control to apply action Δq per step.
- Simulate grasp using a constraint if within `grasp_radius` and “closed”.
- Gymnasium API; wrap via `pufferlib.emulation.GymnasiumPufferEnv` for pufferl.

### Creator pattern (align with mujoco/environment.py)

```python
# environment.py (sketch)
import functools, numpy as np, gymnasium
import pufferlib, pufferlib.emulation
import pybullet as p, pybullet_data

class SOArmBulletGym(gymnasium.Env):
    def __init__(self, urdf_path, render_mode=None, gui=False, max_steps=200,
                 grasp_radius=0.03):
        self.gui = gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        self.arm = p.loadURDF(urdf_path, useFixedBase=True)
        # discover joints and tool link; build obs/action spaces (as in kin env)
        # spawn object cube and goal marker (visual shape + collision shape)
        # store ids; init state; implement reset/step similar to kin env

def single_env_creator(urdf_path, capture_video, gamma, run_name=None, idx=None,
                       obs_norm=True, pufferl=True, render_mode='rgb_array', gui=False, buf=None, seed=0):
    env = SOArmBulletGym(urdf_path=urdf_path, render_mode=render_mode, gui=gui)
    env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    if obs_norm:
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda x: np.clip(x, -10, 10), env.observation_space)
    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    if pufferl: env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return env

def env_creator(urdf_path, gamma=0.99):
    return functools.partial(single_env_creator, urdf_path=urdf_path, capture_video=False, gamma=gamma, pufferl=True)
```

### Notes

- Use GUI only for single‑env eval (slower). For training, use DIRECT mode and, optionally, `RecordVideo` on env 0.
- TCP/gripper: If URDF includes a tool link, use its frame; otherwise apply a fixed transform from flange.
- Tune PD gains for stable position control when applying Δq.

---

## Verification & Tests

1) Kinematics self‑checks:
   - Compare finite‑difference Jacobian columns to `jacobian_pos` (should match by construction).
   - IK convergence: sample random targets inside workspace; ensure residual < threshold in N steps for most cases.

2) Cross‑check FK vs PyBullet:
   - For a few joint configs, set joints in PyBullet and read back EE pose; compare to kin FK (within mm–cm depending on DH rounding).

3) Physics smoke test:
   - Reset → step small Δq → ensure EE position changes; grasp toggles within radius; no simulation errors.

---

## macOS Developer Notes

- PyBullet works on macOS (GUI/DIRECT). For MuJoCo GUI you may need `brew install glfw`.
- Prefer training headless; run a separate eval with GUI to “see it”.
- Use `if __name__ == "__main__"` when using `pufferlib.vector` with multiprocessing on macOS.

---

## Fresh Checkout and Style Audit (local)

- Local PufferLib tree shows envs under `pufferlib/environments/*`, with consistent creator pattern and wrappers. Recent commits indicate fast iteration on envs; code is concise with minimal docstrings.
- We will:
  - Follow the `mujoco/environment.py` pattern for the PyBullet variant.
  - Keep `__init__.py` minimal (`from .environment import *`).
  - Expose knobs via function args; no hardcoded paths.

---

## Ready for Implementation

I am ready to implement both envs with the structure above, plus two examples and optional tests, adhering to the observed PufferLib environment conventions. Once you confirm the URDF path to use from `TheRobotStudio/SO-ARM100` (SO‑101 vs SO‑100) and whether to vendor a small sample DH table, I will:

1) Add `so_arm_kin` (DH) with configurable DH and line‑commented code for clarity.
2) Add `so_arm_bullet` (PyBullet) loading your chosen URDF, with grasp constraint and position control.
3) Add `examples/so_arm_kin_train.py` and `examples/so_arm_bullet_train.py`.
4) Optionally add tests for FK/Jacobian/IK and a smoke test for physics.
5) Run local smoke checks on macOS and prepare the PR following PufferLib’s patterns.

---

## Implementation Status (Updated)

All planned artifacts have been added to the local PufferLib repo:

- Kinematics (no physics):
  - `pufferlib/environments/so_arm_kin/env.py`
    - `DHParams`, `fk_pos`, `jacobian_pos`, `dls_step` helpers.
    - `SOArmKinEnv` PufferEnv: observations `[q, ee, obj, goal, grasped]`, actions `[dq, grip]`.
  - `pufferlib/environments/so_arm_kin/__init__.py`
  - `examples/so_arm_kin_train.py` (PuffeRL trainer using vectorized env)

- PyBullet physics (Option B):
  - `pufferlib/environments/so_arm_bullet/environment.py`
    - `SOArmBulletGym` Gymnasium env loading a provided URDF (SO‑101/100), spawns object + goal, position control on Δq, simple grasp via fixed constraint.
    - `single_env_creator(...)` and `env_creator(...)` matching the mujoco pattern; NormalizeObservation/Reward; optional `RecordVideo` on env 0.
  - `pufferlib/environments/so_arm_bullet/__init__.py`
  - `examples/so_arm_bullet_train.py` (vectorized headless training)
  - `examples/so_arm_bullet_eval_gui.py` (single‑env GUI demo to visualize on macOS)

Notes:
- Kinematics env currently uses placeholder DH values; replace with SO‑101/100 DH to make FK “robot‑correct”.
- Physics env is “robot‑correct” so long as the URDF reflects the real robot (Simulation/ in SO‑ARM100 repo is preferred).

---

## What we implemented end‑to‑end (this repo)

### 1) Environment bindings and interfaces
- Implemented `SOArmBulletGym` as a Gymnasium environment wrapping PyBullet:
  - Loads SO‑ARM100 URDF (`Simulation/SO101/*.urdf`)
  - Discovers joints and their limits; identifies the tool link; exposes `dof` and action/obs spaces.
  - Spawns a cube object and a spherical goal each episode; supports a simple “grasp” via fixed constraint when the gripper (EE) is within `grasp_radius`.
  - Actions: continuous Box `[Δq(DoF), grip]` where `grip>0` closes, `<0` opens. Position control to `q + Δq` with gain/force caps and a few physics substeps.
  - Observations: `[q(DoF), ee(3), obj(3), goal(3), grasped(1)]` as float32.
  - Reward: `-||obj−goal|| + 1*[reached] − 0.001*||Δq||`, termination on `reached` or `max_steps`.

- Puffer wrappers (creator pattern):
  - `single_env_creator(...)` builds the env and applies wrappers:
    - `pufferlib.ClipAction`, `pufferlib.EpisodeStats`
    - `gymnasium.wrappers.NormalizeObservation`, `TransformObservation` (clamp to [-10,10]), `NormalizeReward`, `TransformReward` (clip)
    - Optional `RecordVideo` on env 0
    - Optionally wraps with `pufferlib.emulation.GymnasiumPufferEnv` for compatibility with `pufferlib.vector`
  - `env_creator(urdf_path, gamma)`: `functools.partial(single_env_creator, ...)` – the canonical entrypoint used by vectorization and trainers.

### 2) Vectorization and training integration
- Vectorization: we used `pufferlib.vector.Serial` for single‑env visible training and `Multiprocessing` for throughput. Serial simplifies debugging and avoids macOS multiprocessing concerns.
- Trainer: `pufferlib.pufferl.PuffeRL`, supplied with:
  - `batch_size`, `bptt_horizon`, `minibatch_size`, `max_minibatch_size`
  - Optimizer: switched to `Adam` on Apple Silicon to avoid heavyball’s QR op that’s unsupported by MPS
  - `compile=False` (Torch compile disabled to sidestep MPS op coverage)
  - Automatic mixed precision left off for simplicity on CPU/MPS

### 3) Policy and action distribution
- Continuous actions require a Normal distribution. We updated `examples/so_arm_bullet_train.py`:
  - Policy outputs `μ` (via `mu_head`) and a learned global `log_std` parameter; returns `torch.distributions.Normal(μ, σ)` to `pufferlib.pytorch.sample_logits`, which then samples correctly shaped continuous actions.

### 4) Training/run scripts and modes
- Headless training (PyBullet DIRECT):
  - `examples/so_arm_bullet_train.py /abs/path/to/so101_new_calib.urdf`
  - Creator forced to `gui=False` (DIRECT) to avoid “only one GUI connection” constraints in PyBullet
  - Vectorization: Serial, `num_envs=1`, `batch_size=1` for visible/easy debugging
  - Config fixes: ensure `minibatch_size <= batch_size`; avoid heavyball on MPS; compile disabled
  - Verified logs show training steps, losses, and env stats

- GUI eval (separate process):
  - `examples/so_arm_bullet_eval_gui.py /abs/path/to/so101_new_calib.urdf`
  - Single env in GUI mode; shows a simple scripted controller while printing reward/distance every 50 steps
  - This can run concurrently with headless training to visualize behavior without interfering with the training physics server

### 5) macOS M3 specific adjustments
- Conda env with Python 3.11 (`soarm311`) and `pybullet` from conda‑forge.
- Torch installed from CPU/MPS wheel; `optimizer=adam` and `compile=False` to avoid MPS QR limitations in heavyball.
- Headless training to avoid “one GUI per process” error; GUI kept to separate eval script.

### 6) What connects where
- `examples/so_arm_bullet_train.py`:
  - Builds creator via `single_env_creator(..., gui=False)`
  - `vecenv = pufferlib.vector.make(..., backend=Serial)` – exposes `driver_env`, `single_observation_space`, `single_action_space`
  - Policy is instantiated with `vecenv.driver_env` for shapes
  - `args = pufferl.load_config('default')` → modifies trainer params and instantiates `pufferl.PuffeRL(args['train'], vecenv, policy)`
  - Loop: `evaluate()` → `train()`; dashboard prints environment/timing/loss metrics

---

## Exact install and run (macOS M3)
1) Create conda env and install pybullet/gymnasium/torch:
```
conda create -n soarm311 python=3.11 -y
conda activate soarm311
conda install -y -c conda-forge pybullet gymnasium numpy
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```
2) Install local pufferlib into the conda env without heavy extensions:
```
python -m pip install -e . --no-build-isolation -CNO_OCEAN=1 -CNO_TRAIN=1
```
3) Clone URDF and pick path:
```
git clone https://github.com/TheRobotStudio/SO-ARM100 ~/Documents/Robotics/SO-ARM100
URDF=~/Documents/Robotics/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```
4) Headless training (recommended):
```
python examples/so_arm_bullet_train.py $URDF
```
5) GUI eval (separate terminal):
```
python examples/so_arm_bullet_eval_gui.py $URDF
```

---

## Evaluation UX
- During training, the dashboard prints environment stats (`episode_return`, `dist_to_goal`, `reached`, `grasped`) and PPO style losses.
- For live visualization, run the GUI eval script concurrently; the training instance uses DIRECT mode, so the GUI remains interactive and independent.

---

## Troubleshooting we encountered and fixes applied
- PyBullet build on Python 3.12 → moved to Python 3.11 conda‑forge wheel
- Mixed venv/conda confusion → installed repo editable into the conda env and used conda’s python for all commands
- Action mismatch error → policy changed to Normal distribution for continuous action space
- MPS op (QR) unsupported in heavyball → switched trainer optimizer to Adam and `compile=False`
- One GUI limitation in PyBullet → training in DIRECT mode; GUI used only by eval script
- Gymnasium TransformObservation signature mismatch → use `(env, f)` and keep obs dtype float32

---

## If we re‑implemented from scratch: a faster, cleaner design

Goals
- Maximize sample throughput and stability
- Simple, explicit data flow and vectorization
- Robust macOS support (MPS or CPU), headless by default, GUI in separate process

Key design choices
1) Env core split into model and IO:
   - Physics adapter: PyBullet wrapper that only manages stepping, resetting, and minimal EE/object IO. No shaping inside; just report raw signals (poses, contacts, etc.).
   - Task head: separate reward/termination module with pure‑NumPy transforms and tunable shaping; unit‑testable.
   - Clear action spec: `[Δq, grip]` with per‑joint step clamp at the env boundary to prevent sim instabilities.

2) Vectorization default = Multiprocessing for throughput, but a reliable Serial fallback. Add config gate to choose `num_envs`, `num_workers`, `batch_size` automatically from CPU cores.

3) Policy/runtime:
   - Always model actions with Normal distribution for continuous Box; learn per‑dimension log_std.
   - Pure Adam defaults (compile off) on macOS; enable compile and heavier optimizers (heavyball) on Linux/CUDA.

4) Observation pipeline:
   - Normalize + clamp in one place; ensure dtype=float32 strictly; avoid accidental dtype swaps.

5) Diagnostics:
   - Smoke test that compares FK from URDF (PyBullet) against the kinematics module (SO100 DH) for a few joint configs.
   - Unit tests for reward components and grasp toggling.

6) GUI strategy:
   - Training runs DIRECT only; GUI eval is an explicit, separate process. Optionally record MP4 on env 0.

Why this is better
- Clear boundaries (physics vs task) make reward tuning and debugging simpler.
- Normalized continuous actions remove mismatches and failure cases with discrete sampling.
- DIRECT/GUI separation avoids PyBullet connection errors and improves stability.
- Adam baseline + compile off on MPS eliminates backend incompatibilities; yet the same code can flip to CUDA/compile on Linux.

Implementation sketch
```
# env/__init__.py
def make_creator(urdf_path, gamma=0.99, gui=False):
    return functools.partial(single_env_creator, urdf_path=urdf_path, gamma=gamma, gui=gui, pufferl=True)

# env/environment.py
class SOArmBulletGym(gym.Env):
    # minimal physics wrapper with explicit step clamp and IO

def single_env_creator(...):
    env = SOArmBulletGym(...)
    env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    env = Normalize/Transform obs+reward
    return pufferlib.emulation.GymnasiumPufferEnv(env)

# examples/train.py
creator = make_creator(urdf, gui=False)
vecenv = pufferlib.vector.make(creator, num_envs=N, num_workers=W, batch_size=B, backend=pufferlib.vector.Multiprocessing)
policy = MLPNormal(vecenv.driver_env)
args = pufferl.load_config('default'); patch for Adam/MPS and minibatches
trainer = pufferl.PuffeRL(args['train'], vecenv, policy)
while ...: trainer.evaluate(); trainer.train()

# examples/eval_gui.py
creator = make_creator(urdf, gui=True)
env = creator()
scripted or trained policy rollout with visuals
```

Expected speedups
- Multiprocessing with sensible `num_envs/num_workers/batch_size` matching core count.
- Minimal per‑step Python overhead (task shaping in NumPy, continuous actions sampled efficiently).
- Avoidance of GUI overhead during training.

---

## TL;DR runbook
- Train (headless):
```
python examples/so_arm_bullet_train.py /path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```
- Visualize:
```
python examples/so_arm_bullet_eval_gui.py /path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```
- Customize trainer: edit `examples/so_arm_bullet_train.py` minibatches/batch size/optimizer.


---

## How to Run (Option B: PyBullet)

1) Install PyBullet (macOS):
   - Recommended: Python 3.11 (prebuilt wheels available)
     - `python3.11 -m venv .venv311 && source .venv311/bin/activate`
     - `pip install -U pip setuptools wheel`
     - `pip install pybullet`
   - Or conda‑forge:
     - `conda create -n soarm python=3.11 && conda activate soarm`
     - `conda install -c conda-forge pybullet`

   Troubleshooting build from source on macOS:
   - If you see `error: command '/usr/bin/clang' failed with exit code 1`, you’re building from source (likely on Python 3.12). Prefer Python 3.11 or install Xcode CLT and try a version pin (e.g., `pip install "pybullet==3.2.6"`).

2) Choose the URDF:
   - From `TheRobotStudio/SO-ARM100` repo, pick the SO‑101/100 URDF under `Simulation/` (preferred) or a cleaned URDF under `URDF/`.
   - Example: `/abs/path/to/SO-ARM100/Simulation/SO101.urdf`

3) Train (headless vectorized):
   - `python examples/so_arm_bullet_train.py /abs/path/to/SO101.urdf`

4) Visualize (GUI):
   - `python examples/so_arm_bullet_eval_gui.py /abs/path/to/SO101.urdf`
   - GUI is single‑env, slower but interactive. Use headless training for throughput.

5) Optional videos:
   - The creator supports `RecordVideo` on env 0 (mirrors mujoco pattern). Wire a `capture_video=True` flag in your launcher if you want MP4 artifacts during training.

---

## How to Run (Kinematics, no physics)

- `python examples/so_arm_kin_train.py`
- Replace the placeholder DH in `SOArmKinEnv` constructor with your SO‑101/100 DH to make FK/IK exact.

---

## Verification & Next Steps (Updated)

- FK/Jacobian/IK tests (optional to add next):
  - Jacobian finite‑difference self‑check.
  - IK Monte‑Carlo convergence inside workspace.
  - Cross‑check FK vs PyBullet EE pose for a few joint configs.

- URDF specifics:
  - If the URDF doesn’t name the tool link with `tool/tcp/ee`, the env defaults to the last link as EE; we can pin a specific link if desired.

- DH integration:
  - Provide numeric DH table for SO‑101/100; I’ll set as defaults in `so_arm_kin` and add minimal tests.
