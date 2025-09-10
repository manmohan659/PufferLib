## SO‑101 Arm in PufferLib — Implemented, how to run, and engineering journal

This document reflects the current code in this branch. It replaces prior plans and removes unused or conflicting sections.

## What we implemented (current state)

- C‑native SO‑ARM environment (fast, CPU‑only, no external physics)
  - Files: `pufferlib/ocean/so_arm/so_arm.h`, `pufferlib/ocean/so_arm/binding.c`, `pufferlib/ocean/so_arm/so_arm.py`, `pufferlib/ocean/so_arm/__init__.py`
  - Features:
    - 6‑DoF control: 5 arm joints + wrist (“neck”) rotation θ6, plus a gripper scalar in actions
    - Denavit–Hartenberg FK with SO100/SO101‑style parameters and mech↔DH mapping
    - Zero‑alloc vectorized stepping via ocean `env_binding.h` (observations/actions/rewards in preallocated NumPy buffers)
    - Task: pick a red ball and place it into a cyan bin; radius‑aware grasp; success on placement or timeout
    - Minimal Raylib 3D renderer in `c_render`: links, visible gripper pads, red ball, cyan bin
    - Robust object handling: keep‑out spawn near links; push‑out if sphere overlaps links
    - Floor safety: backtracks joint step if any link would penetrate z<0

- Scripted demo (for sanity and visualization)
  - File: `examples/so_arm_c_scripted_demo.py`
  - Damped‑least‑squares IK controller: go to ball → close → go to bin → open (5400 steps ~90s)

- PPO example for C‑native env
  - File: `examples/so_arm_c_train.py`
  - Continuous Normal policy with Adam, `pufferlib.vector`. Auto‑selects device (CUDA/MPS/CPU)

- PyBullet visual env (optional, for physics demo)
  - Files: `pufferlib/environments/so_arm_bullet/environment.py`, `pufferlib/environments/so_arm_bullet/__init__.py`
  - Loads URDF, mirrors the C task (reset/step/observe). Not required for core training

- Supportive edits
  - `pufferlib/pufferl.py`: optional `rich`/`rich_argparse` fallback
  - `pufferlib/spaces.py`: optional `gym` import; we use Gymnasium
  - `pufferlib/ocean/environment.py`: register `so_arm`
  - `tests/test_so_arm_c_smoke.py`: smoke import + reset/step

## Interfaces (ground truth)

- Observation: `[q(dof), ee(3), obj(3), goal(3), grasped(1)]` (float32)
- Action: `Δq(dof)` per step, plus `grip∈[-1,1]` (close>0, open<0)
- Reward: `+ shaping(approach) − 0.001*||Δq|| + bonuses (grasp/place)`; terminate on place or timeout
- Defaults:
  - `dof=6` (5 arm joints + wrist)
  - `grasp_radius≈0.03`, `max_steps≈200`, workspace bounds configurable in `SoArm(...)`
  - Gripper: `jaw_min/jaw_max/grip_speed` exposed; object `obj_radius` configurable

## New parameters and calibration (wrapper)

The Python wrapper accepts optional realism parameters and a calibration file:

- `servo_tau`: per‑joint first‑order lag (steps); `0.0` disables
- `joint_rate`: per‑joint max delta per step (rad/step)
- `jaw_min`, `jaw_max`, `grip_speed`: parallel‑jaw model parameters (meters)
- `ee_radius`: optional EE marker radius for render
- `calibration`: dict or YAML path to override DH/limits/servo/gripper

YAML example:
```yaml
dh:
  a: [0.0304, 0.116, 0.1347, 0.0, 0.0, 0.0]
  alpha: [1.570796, 0.0, 0.0, -1.570796, 0.0, 0.0]
  d: [0.0542, 0.0, 0.0, 0.0, 0.0609, 0.0]
joint_min: [-2.2, -3.1416, -1.0, -3.1416, -3.1416, -3.1416]
joint_max: [ 2.2,  0.2,      3.1416,  1.8,   3.1416,  3.1416]
servo_tau: [0.0, 1.5, 1.2, 1.0, 0.8, 0.5]
joint_rate: [0.05, 0.04, 0.04, 0.05, 0.05, 0.08]
jaw: {min: 0.0, max: 0.04, speed: 0.004}
```

## How to set up and run

### One‑time setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install gymnasium numpy
pip install -e .
```

macOS (Raylib dylib on PATH) if needed:
```bash
export DYLD_LIBRARY_PATH="$PWD/raylib-5.5_macos/lib:$DYLD_LIBRARY_PATH"
```

### Visual scripted pick‑and‑place (C‑native, Raylib window)
```bash
PYTHONNOUSERSITE=1 PYTHONPATH=. python examples/so_arm_c_scripted_demo.py
```
Tip: don’t pipe output (e.g., through `sed`) or the process may end before you can view the window. To keep it running, launch it in a dedicated terminal.

### Train the C‑native env (fast)
```bash
# Short smoke
PYTHONNOUSERSITE=1 PYTHONPATH=. python examples/so_arm_c_train.py

# Longer run (e.g., 10M steps)
PYTHONNOUSERSITE=1 PYTHONPATH=. \
  PUFFER_TRAIN_total_timesteps=10000000 \
  PUFFER_TRAIN_batch_size=8192 \
  PUFFER_TRAIN_bptt_horizon=256 \
  PUFFER_TRAIN_minibatch_size=1024 \
  python examples/so_arm_c_train.py
```

### Optional: PyBullet physics demo
```bash
pip install pybullet
# Headless train demo
PYTHONNOUSERSITE=1 PYTHONPATH=. python examples/so_arm_bullet_train.py /abs/path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
# GUI eval
PYTHONNOUSERSITE=1 PYTHONPATH=. python examples/so_arm_bullet_eval_gui.py /abs/path/to/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
```

## Rendering and realism

- Renderer shows links, EE sphere (optional), two visible pads, red ball, cyan bin
- Grasp is radius‑aware and uses jaw width to hold/release the ball
- Spawn has keep‑out around links; step has push‑out to avoid overlaps; floor backtracking prevents z<0
- Wrist joint (θ6) active; jaw pads reflect opening width

## Design rationale

- Keep training fast and portable: C‑native kinematics, preallocated buffers, inline renderer
- Use PyBullet only for demos/validation, not as a hard dependency
- Header‑inline C keeps ocean build simple and robust on macOS/Clang

## Files changed

- Added: `pufferlib/ocean/so_arm/{so_arm.h,binding.c,so_arm.py,__init__.py}`
- Added: `examples/{so_arm_c_train.py,so_arm_c_scripted_demo.py}`
- Added: `pufferlib/environments/so_arm_bullet/{environment.py,__init__.py}` (optional)
- Added: `tests/test_so_arm_c_smoke.py`
- Updated: `pufferlib/ocean/environment.py` (register), `pufferlib/pufferl.py` (rich fallback), `pufferlib/spaces.py` (gym optional)
- Updated (latest):
  - `pufferlib/ocean/so_arm/so_arm.py`: added `calibration`, `servo_tau`, `joint_rate`, `jaw_*`, `grip_speed`, `ee_radius`; relative import of `binding`
  - `pufferlib/ocean/so_arm/__init__.py`: lazy imports via `__getattr__` to prevent circular import
  - `pufferlib/ocean/so_arm/so_arm.h`: added servo model, jaw state, keep‑out/push‑out, floor backtracking; fixed logging bug in `c_step`

## Engineering journal (how we got here)

- Consolidated C to header‑inline (`so_arm.h`) after a macOS dlopen symbol error (`_c_close`)
- Implemented FK, workspace bounds, object spawn keep‑out, grasp/place logic, Raylib renderer
- Added jaw model and gripper pads; made sphere radius configurable; added push‑out when overlapping links
- Extended Python wrapper to 6‑DoF; added servo rate/lag, gripper params, and YAML calibration loader
- Fixed circular import by switching to relative import in `so_arm.py` and lazy `__getattr__` in `__init__.py`
- Diagnosed Raylib window “open/close instantly” when piping output; recommend running directly in a terminal
- Reproduced a segmentation fault; root cause: `c_step` logged `dist`/`reached` (undefined). Fix: use `obj_to_goal` and explicit success flag. Verified Raylib window renders and demo completes
- Added macOS note for `DYLD_LIBRARY_PATH` if Raylib dylib is not found

## Troubleshooting

- ImportError (circular import): use lazy `__getattr__` in `__init__.py` and relative `from . import binding`
- Segmentation fault on step: ensure `so_arm.h` uses defined values in logging; updated to `obj_to_goal` and success flag
- `ModuleNotFoundError: gym`: we use Gymnasium; `pufferlib/spaces.py` makes `gym` optional
- `TypeError` from `rich`: `pufferlib/pufferl.py` now has a robust fallback
- Raylib dylib on macOS: set `DYLD_LIBRARY_PATH="$PWD/raylib-5.5_macos/lib:$DYLD_LIBRARY_PATH"`

## Next steps (optional)

- Tune DH and limits from measured SO‑101 data; expose wrist/tool transforms
- Add analytic Jacobian and optional small IK step in C for demos
- Stage reward (reach→grasp→transport→place) and light domain randomization (obj size, lag)

## Long runs and H100 guidance

H100 setup (CUDA):
```bash
python -m pip install -U pip setuptools wheel gymnasium
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Launch a long C‑native run (GPU for policy):
```bash
export CUDA_VISIBLE_DEVICES=0
PYTHONNOUSERSITE=1 PYTHONPATH=. \
  PUFFER_TRAIN_total_timesteps=50000000 \
  PUFFER_TRAIN_batch_size=65536 \
  PUFFER_TRAIN_bptt_horizon=256 \
  PUFFER_TRAIN_minibatch_size=4096 \
  python examples/so_arm_c_train.py | tee runs/soarm_c_native_h100.log
```

Tips:
- Increase `num_envs/num_workers` in `examples/so_arm_c_train.py` (e.g., 64/16) to keep the GPU fed
- Maintain `segments = batch_size / bptt_horizon ≥ total_agents`
- Add periodic checkpoints by calling `trainer.save()` every N steps