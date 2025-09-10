import numpy as np
import gymnasium

import pufferlib
from . import binding


class SoArm(pufferlib.PufferEnv):
    def __init__(self,
                 num_envs=1,
                 dof=6,
                 calibration=None,
                 dh_a=None,
                 dh_alpha=None,
                 dh_d=None,
                 joint_min=None,
                 joint_max=None,
                 servo_tau=None,
                 joint_rate=None,
                 jaw_min=0.0,
                 jaw_max=0.04,
                 grip_speed=0.004,
                 ee_radius=0.0,
                 ws_min=(-0.6, -0.6, 0.0),
                 ws_max=(0.6, 0.6, 0.6),
                 max_steps=200,
                 grasp_radius=0.03,
                 obj_radius=0.02,
                 buf=None,
                 seed=0):
        self.num_agents = num_envs
        self.dof = dof

        # If a calibration dict or YAML path is provided, load parameters first
        if calibration is not None:
            cfg = None
            if isinstance(calibration, dict):
                cfg = calibration
            else:
                try:
                    import os, yaml  # type: ignore
                    if isinstance(calibration, str) and os.path.exists(calibration):
                        with open(calibration, 'r') as f:
                            cfg = yaml.safe_load(f)
                except Exception:
                    cfg = None
            if isinstance(cfg, dict):
                dh = cfg.get('dh', {})
                dh_a = dh.get('a', dh_a)
                dh_alpha = dh.get('alpha', dh_alpha)
                dh_d = dh.get('d', dh_d)
                joint_min = cfg.get('joint_min', joint_min)
                joint_max = cfg.get('joint_max', joint_max)
                servo_tau = cfg.get('servo_tau', servo_tau)
                joint_rate = cfg.get('joint_rate', joint_rate)
                jaw = cfg.get('jaw', {})
                jaw_min = jaw.get('min', jaw_min)
                jaw_max = jaw.get('max', jaw_max)
                grip_speed = jaw.get('speed', grip_speed)

        # Defaults to SO100/SO101 DH (theta is variable): rows [theta, d, a, alpha]
        # We pass a/alpha/d separately to C, in Craig's convention.
        if dh_a is None:
            dh_a = [0.0304, 0.116, 0.1347, 0.0, 0.0, 0.0][:dof]
        if dh_alpha is None:
            dh_alpha = [np.pi/2, 0.0, 0.0, -np.pi/2, 0.0, 0.0][:dof]
        if dh_d is None:
            dh_d = [0.0542, 0.0, 0.0, 0.0, 0.0609, 0.0][:dof]

        # Derive DH joint limits from mechanical limits via from_mech_to_dh map
        if joint_min is None or joint_max is None:
            beta = np.deg2rad(14.45)
            mech_low = np.array([-2.2, -np.pi, 0.0, -2.0, -np.pi, -0.2], dtype=np.float64)
            mech_up  = np.array([ 2.2,  0.2,  np.pi, 1.8,  np.pi,  2.0], dtype=np.float64)
            # mapping for first 5 joints
            def map_dh(qm):
                qd = np.empty(5, dtype=np.float64)
                qd[0] = qm[0]
                qd[1] = -qm[1] - beta
                qd[2] = -qm[2] + beta
                qd[3] = -qm[3] - np.pi/2
                qd[4] = -qm[4] - np.pi/2
                return qd
            mins = []; maxs = []
            for i in range(min(5, dof)):
                a = mech_low.copy(); b = mech_up.copy()
                da = map_dh(a)[i]; db = map_dh(b)[i]
                lo, hi = (da, db) if da <= db else (db, da)
                mins.append(lo); maxs.append(hi)
            # If a 6th wrist joint is present, allow full roll range
            if dof > 5:
                mins.append(-np.pi)
                maxs.append( np.pi)
            joint_min = mins[:dof]
            joint_max = maxs[:dof]

        # Defaults for servo model if not provided
        if servo_tau is None:
            # 0 => no lag; you can pass measured values via YAML later
            servo_tau = [0.0] * dof
        if joint_rate is None:
            # Conservative per-step joint deltas; can be tuned from measured speeds
            joint_rate = [0.05] * dof

        obs_dim = dof + 3 + 3 + 3 + 1
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=np.array([-0.05]*dof + [-1], np.float32),
            high=np.array([ 0.05]*dof + [ 1], np.float32),
            dtype=np.float32,
        )

        super().__init__(buf)
        # Actions buffer must be float32 contiguous
        self.actions = np.zeros((num_envs, self.single_action_space.shape[0]), dtype=np.float32)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            dof=int(dof),
            max_steps=int(max_steps),
            grasp_radius=float(grasp_radius),
            obj_radius=float(obj_radius),
            dh_a=list(dh_a),
            dh_alpha=list(dh_alpha),
            dh_d=list(dh_d),
            joint_min=list(joint_min),
            joint_max=list(joint_max),
            servo_tau=list(servo_tau),
            joint_rate=list(joint_rate),
            jaw_min=float(jaw_min),
            jaw_max=float(jaw_max),
            grip_speed=float(grip_speed),
            ee_radius=float(ee_radius),
            ws_min=list(ws_min),
            ws_max=list(ws_max),
        )

    def reset(self, seed=None):
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, int(seed))
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


