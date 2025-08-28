from __future__ import annotations

import functools
import os
from typing import Optional

import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation


try:
    import pybullet as p
    import pybullet_data
except Exception as e:  # pragma: no cover - optional dependency
    p = None


class SOArmBulletGym(gymnasium.Env):
    """PyBullet-based SO-ARM env (URDF required).

    Observation/Action layout matches the kinematics env for parity.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        urdf_path: str,
        render_mode: Optional[str] = None,
        gui: bool = False,
        max_steps: int = 200,
        grasp_radius: float = 0.03,
    ):
        if p is None:
            raise ImportError("pybullet is required for SOArmBulletGym. Install with `pip install pybullet`. ")

        self.render_mode = render_mode
        self.gui = bool(gui)
        self.cid = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load plane for reference and the arm URDF
        self.plane = p.loadURDF("plane.urdf")
        if not os.path.isabs(urdf_path):
            # If relative, allow using current working directory
            urdf_path = os.path.abspath(urdf_path)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.arm = p.loadURDF(urdf_path, useFixedBase=True)

        # Discover revolute joints and try to find a tool/ee link
        self.joints = []
        self.joint_min, self.joint_max = [], []
        self.tool_link = p.getNumJoints(self.arm) - 1
        for j in range(p.getNumJoints(self.arm)):
            info = p.getJointInfo(self.arm, j)
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE or jtype == p.JOINT_PRISMATIC:
                self.joints.append(j)
                lim_low, lim_high = info[8], info[9]
                # If limits are invalid, fall back to wide default
                if lim_low > lim_high:
                    lim_low, lim_high = -np.pi, np.pi
                self.joint_min.append(lim_low)
                self.joint_max.append(lim_high)
            name = info[12].decode("utf-8") if isinstance(info[12], (bytes, bytearray)) else str(info[12])
            if any(k in name.lower() for k in ("tool", "tcp", "ee")):
                self.tool_link = j

        self.joints = np.array(self.joints, dtype=int)
        self.joint_min = np.array(self.joint_min, dtype=float)
        self.joint_max = np.array(self.joint_max, dtype=float)
        self.dof = len(self.joints)

        # Object and goal
        self.obj_sid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
        self.obj_vid = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02], rgbaColor=[0.9, 0.3, 0.3, 1])
        self.goal_sid = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
        self.goal_vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0.3, 0.9, 0.3, 1])

        self.obj_id = None
        self.goal_id = None
        self.grasp_cid = None
        self.max_steps = int(max_steps)
        self.grasp_radius = float(grasp_radius)

        # Observation/action spaces
        obs_low = np.concatenate([
            self.joint_min,
            [-1, -1, 0],  # ee (coarse bounds)
            [-1, -1, 0],  # obj
            [-1, -1, 0],  # goal
            [0.0],
        ]).astype(np.float32)
        obs_high = np.concatenate([
            self.joint_max,
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1.0],
        ]).astype(np.float32)
        self.observation_space = gymnasium.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-0.05] * self.dof + [-1.0], dtype=np.float32),
            high=np.array([0.05] * self.dof + [1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.t = 0
        self.grasped = False
        self.q = np.zeros(self.dof, dtype=float)
        self._reset_bodies()

    # --- helpers ---
    def _reset_bodies(self):
        # Reset joints
        for i, j in enumerate(self.joints):
            p.resetJointState(self.arm, int(j), 0.0)
            self.q[i] = 0.0
        # Spawn object and goal at random positions
        rng = np.random.default_rng()
        obj_pos = rng.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.25])
        goal_pos = rng.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.25])
        if self.obj_id is not None:
            p.removeBody(self.obj_id)
        if self.goal_id is not None:
            p.removeBody(self.goal_id)
        self.obj_id = p.createMultiBody(baseMass=0.05, baseCollisionShapeIndex=self.obj_sid,
                                        baseVisualShapeIndex=self.obj_vid, basePosition=obj_pos.tolist())
        self.goal_id = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=self.goal_sid,
                                         baseVisualShapeIndex=self.goal_vid, basePosition=goal_pos.tolist())
        if self.grasp_cid is not None:
            p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None

    def _get_ee_pos(self) -> np.ndarray:
        ls = p.getLinkState(self.arm, int(self.tool_link), computeForwardKinematics=True)
        return np.array(ls[0], dtype=float)

    def _get_body_pos(self, bid) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(bid)[0], dtype=float)

    def _obs(self):
        ee = self._get_ee_pos()
        obj = self._get_body_pos(self.obj_id)
        goal = self._get_body_pos(self.goal_id)
        g = 1.0 if self.grasped else 0.0
        return np.concatenate([self.q, ee, obj, goal, [g]]).astype(np.float32)

    # --- gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.grasped = False
        self._reset_bodies()
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=float)
        dq = a[: self.dof]
        grip = float(a[self.dof])

        # Position control to target q (current + dq), clamped to limits
        q_target = np.clip(self.q + dq, self.joint_min, self.joint_max)
        for i, j in enumerate(self.joints):
            p.setJointMotorControl2(
                bodyIndex=self.arm,
                jointIndex=int(j),
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(q_target[i]),
                positionGain=0.5,
                velocityGain=0.5,
                force=5.0,
            )
        # Step simulation a few substeps for stability
        for _ in range(4):
            p.stepSimulation()
        self.q = q_target.copy()

        ee = self._get_ee_pos()
        obj = self._get_body_pos(self.obj_id)
        goal = self._get_body_pos(self.goal_id)

        # Grasp logic via constraint
        if grip > 0.0 and not self.grasped:
            if np.linalg.norm(ee - obj) <= self.grasp_radius:
                self.grasp_cid = p.createConstraint(
                    parentBodyUniqueId=self.arm,
                    parentLinkIndex=int(self.tool_link),
                    childBodyUniqueId=self.obj_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 1],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                )
                self.grasped = True
        elif grip < 0.0 and self.grasped:
            if self.grasp_cid is not None:
                p.removeConstraint(self.grasp_cid)
                self.grasp_cid = None
            self.grasped = False

        # Reward and termination
        dist = float(np.linalg.norm(obj - goal))
        reached = dist <= 0.03
        reward = -dist + (1.0 if reached else 0.0) - 0.001 * float(np.linalg.norm(dq))

        self.t += 1
        terminated = bool(reached)
        truncated = bool(self.t >= self.max_steps)
        obs = self._obs()
        info = {"dist_to_goal": dist, "reached": bool(reached), "grasped": bool(self.grasped)}
        return obs, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - optional
        if self.render_mode != "rgb_array":
            return None
        # Use TinyRenderer camera; for brevity return None here
        return None

    def close(self):  # pragma: no cover - optional
        try:
            if p and self.cid is not None:
                p.disconnect(self.cid)
        except Exception:
            pass


def single_env_creator(
    urdf_path: str,
    capture_video: bool,
    gamma: float,
    run_name: str | None = None,
    idx: int | None = None,
    obs_norm: bool = True,
    pufferl: bool = True,
    render_mode: str = "rgb_array",
    gui: bool = False,
    buf=None,
    seed: int = 0,
):
    env = SOArmBulletGym(urdf_path=urdf_path, render_mode=render_mode, gui=gui)
    if capture_video and (idx == 0) and run_name is not None:
        env = gymnasium.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    if obs_norm:
        env = gymnasium.wrappers.NormalizeObservation(env)
        # Ensure dtype stays float32 for emulation checks
        env = gymnasium.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10).astype(np.float32))
    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
    if pufferl:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return env


def env_creator(urdf_path: str, gamma: float = 0.99):
    """Return a partial that creates a wrapped Gym env compatible with pufferl."""
    return functools.partial(
        single_env_creator,
        urdf_path=urdf_path,
        capture_video=False,
        gamma=gamma,
        pufferl=True,
    )
