import functools
import numpy as np
import gymnasium

import pufferlib
import pufferlib.emulation


def _try_import_pybullet():
    try:
        import pybullet as p
        import pybullet_data  # noqa: F401
        return p
    except Exception as e:
        raise ImportError(
            "PyBullet is required for so_arm_bullet. Install with 'pip install pybullet' or conda-forge."
        ) from e


class SOArmBulletGym(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, urdf_path, render_mode=None, gui=False, max_steps=200, grasp_radius=0.03,
                 obj_radius=0.02, obj_mass=0.2):
        super().__init__()
        self.p = _try_import_pybullet()
        self.gui = gui
        self.cid = self.p.connect(self.p.GUI if gui else self.p.DIRECT)
        self.p.setAdditionalSearchPath("pybullet_data")
        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(1.0 / 240.0)

        self.arm = self.p.loadURDF(urdf_path, useFixedBase=True)

        # Discover joints (revolute and prismatic) and EE link
        self.joints = []
        self.joint_lower = []
        self.joint_upper = []
        num_j = self.p.getNumJoints(self.arm)
        self.ee_link = -1
        for j in range(num_j):
            ji = self.p.getJointInfo(self.arm, j)
            jtype = ji[2]
            if jtype in (self.p.JOINT_REVOLUTE, self.p.JOINT_PRISMATIC):
                self.joints.append(j)
                self.joint_lower.append(ji[8])
                self.joint_upper.append(ji[9])
            name = ji[12].decode("utf-8") if isinstance(ji[12], (bytes, bytearray)) else str(ji[12])
            if any(k in name.lower() for k in ("tool", "tcp", "ee", "gripper")):
                self.ee_link = j
        if self.ee_link < 0:
            self.ee_link = num_j - 1 if num_j > 0 else -1

        self.dof = len(self.joints)
        self.max_steps = int(max_steps)
        self.grasp_radius = float(grasp_radius)
        self.obj_radius = float(obj_radius)
        self.obj_mass = float(obj_mass)

        # Observation and action spaces
        obs_dim = self.dof + 3 + 3 + 3 + 1
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=np.array([-0.05] * self.dof + [-1], dtype=np.float32),
            high=np.array([0.05] * self.dof + [1], dtype=np.float32),
            dtype=np.float32,
        )

        # State
        self.q = np.zeros(self.dof, dtype=np.float32)
        self.ee = np.zeros(3, dtype=np.float32)
        self.obj = np.zeros(3, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.grasped = False
        self.grasp_cid = None
        self.t = 0

        # Object and goal placeholders
        self.obj_id = None
        self.goal_vis_id = None
        self.render_mode = render_mode

    def _spawn_world(self):
        if self.obj_id is not None:
            self.p.removeBody(self.obj_id)
            self.obj_id = None
        if self.goal_vis_id is not None:
            self.p.removeBody(self.goal_vis_id)
            self.goal_vis_id = None

        # Small sphere as object (radius, mass configurable)
        col = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=self.obj_radius)
        vis = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=self.obj_radius, rgbaColor=[1, 0, 0, 1])
        self.obj_id = self.p.createMultiBody(baseMass=float(self.obj_mass), baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                             basePosition=self.obj.tolist())

        # Sphere visual for goal
        vis2 = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=max(0.02, self.obj_radius*1.2), rgbaColor=[0, 1, 1, 0.7])
        self.goal_vis_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis2,
                                                  basePosition=self.goal.tolist())

    def _ee_pos(self):
        if self.ee_link >= 0:
            ls = self.p.getLinkState(self.arm, self.ee_link, computeForwardKinematics=True)
            return np.array(ls[0], dtype=np.float32)
        else:
            base = self.p.getBasePositionAndOrientation(self.arm)[0]
            return np.array(base, dtype=np.float32)

    def _obs(self):
        return np.concatenate([self.q, self.ee, self.obj, self.goal, [1.0 if self.grasped else 0.0]]).astype(np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Random small q and set joint states
        self.q = np.random.uniform(-0.1, 0.1, size=self.dof).astype(np.float32)
        for qi, j in zip(self.q, self.joints):
            self.p.resetJointState(self.arm, j, float(qi))
        self.ee = self._ee_pos()
        # Workspace sampling
        ws_min = np.array([-0.5, -0.5, 0.05], dtype=np.float32)
        ws_max = np.array([0.5, 0.5, 0.6], dtype=np.float32)
        self.obj = np.random.uniform(ws_min, ws_max).astype(np.float32)
        self.goal = np.random.uniform(ws_min, ws_max).astype(np.float32)
        self.grasped = False
        self.t = 0
        self._spawn_world()
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        dq, grip = a[: self.dof], float(a[self.dof])
        target_q = np.clip(self.q + dq, self.joint_lower, self.joint_upper)
        # position control
        for j, tq in zip(self.joints, target_q):
            self.p.setJointMotorControl2(self.arm, j, self.p.POSITION_CONTROL, targetPosition=float(tq), positionGain=0.4, force=50)
        for _ in range(8):
            self.p.stepSimulation()
        self.q = target_q
        self.ee = self._ee_pos()

        # grasp/open
        if grip > 0 and not self.grasped:
            if np.linalg.norm(self.ee - self.obj) <= (self.grasp_radius + self.obj_radius):
                # Create fixed constraint
                self.grasp_cid = self.p.createConstraint(
                    self.arm,
                    self.ee_link if self.ee_link >= 0 else -1,
                    self.obj_id,
                    -1,
                    self.p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                )
                self.grasped = True
        if grip < 0 and self.grasped:
            if self.grasp_cid is not None:
                self.p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None
            self.grasped = False

        # Reward
        dist = float(np.linalg.norm(self.obj - self.goal))
        reached = dist <= 0.03
        r = -dist + (1.0 if reached else 0.0) - 0.001 * float(np.linalg.norm(dq))
        self.t += 1
        term = bool(reached)
        trunc = bool(self.t >= self.max_steps)
        return self._obs(), r, term, trunc, {
            "dist_to_goal": dist,
            "reached": reached,
            "grasped": bool(self.grasped),
        }

    def render(self):
        # GUI handled by PyBullet viewer
        pass

    def close(self):
        try:
            self.p.disconnect(self.cid)
        except Exception:
            pass


def single_env_creator(urdf_path, capture_video, gamma, run_name=None, idx=None, obs_norm=True, pufferl=True, render_mode='rgb_array', gui=False, buf=None, seed=0):
    env = SOArmBulletGym(urdf_path=urdf_path, render_mode=render_mode, gui=gui)
    env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    if obs_norm:
        env = gymnasium.wrappers.NormalizeObservation(env)
        env = gymnasium.wrappers.TransformObservation(env, lambda x: np.clip(x, -10, 10), env.observation_space)
    env = gymnasium.wrappers.NormalizeReward(env, gamma=gamma)
    env = gymnasium.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
    if pufferl:
        env = pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
    return env


def env_creator(urdf_path, gamma=0.99):
    return functools.partial(single_env_creator, urdf_path=urdf_path, capture_video=False, gamma=gamma, pufferl=True)
