import numpy as np
import gymnasium

import pufferlib
from . import binding


class SoArm(pufferlib.PufferEnv):
    def __init__(self,
                 num_envs=1,
                 dof=6,
                 calibration=None,
                 gripper_first=True,
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
        self.gripper_first = bool(gripper_first)

        # If a calibration dict or YAML path is provided, load parameters first
        if calibration is not None:
            cfg = None
            if isinstance(calibration, dict):
                cfg = calibration
            else:
                try:
                    import os
                    if isinstance(calibration, str) and os.path.exists(calibration):
                        ext = os.path.splitext(calibration)[1].lower()
                        with open(calibration, 'r') as f:
                            if ext == '.json':
                                import json
                                cfg = json.load(f)
                            else:
                                try:
                                    import yaml  # type: ignore
                                    cfg = yaml.safe_load(f)
                                except Exception:
                                    # Fallback: attempt JSON parse
                                    f.seek(0)
                                    import json
                                    cfg = json.load(f)
                except Exception:
                    cfg = None
            if isinstance(cfg, dict):
                dh = cfg.get('dh', {})
                dh_a = dh.get('a', dh_a)
                dh_alpha = dh.get('alpha', dh_alpha)
                dh_d = dh.get('d', dh_d)
                # direct DH limits (radians) take precedence
                joint_min = cfg.get('joint_min', joint_min)
                joint_max = cfg.get('joint_max', joint_max)
                # Map LeRobot sim_limits joint_limits_deg (mechanical, degrees) to DH radians
                limits_deg = cfg.get('joint_limits_deg')
                if limits_deg and (joint_min is None or joint_max is None):
                    import math
                    # normalize keys like "shoulder_pan(6th)" -> "shoulder_pan"
                    def knorm(k: str) -> str:
                        return k.split('(')[0].strip()
                    # order: prefer provided sampling_order (excluding gripper), else default
                    names = []
                    if isinstance(cfg.get('sampling_order'), (list, tuple)):
                        for n in cfg['sampling_order']:
                            nn = str(n)
                            if knorm(nn) != 'gripper':
                                names.append(knorm(nn))
                    if not names:
                        names = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']
                    beta = math.radians(14.45)
                    qmin = [None]*self.dof
                    qmax = [None]*self.dof
                    for i, name in enumerate(names[:min(5, self.dof)]):
                        lim = limits_deg.get(name) or limits_deg.get(f"{name}")
                        if not lim:
                            # also try key variants with parentheses
                            for k, v in limits_deg.items():
                                if knorm(str(k)) == name:
                                    lim = v; break
                        if not lim:
                            continue
                        lo = math.radians(float(lim['min']))
                        hi = math.radians(float(lim['max']))
                        # mechanical -> DH mapping
                        def mech_to_dh(idx: int, x: float) -> float:
                            if idx == 0:  # shoulder_pan
                                return x
                            if idx == 1:  # shoulder_lift
                                return -x - beta
                            if idx == 2:  # elbow_flex
                                return -x + beta
                            if idx == 3:  # wrist_flex
                                return -x - math.pi/2
                            if idx == 4:  # wrist_roll
                                return x
                            return x
                        lo_m = mech_to_dh(i, lo)
                        hi_m = mech_to_dh(i, hi)
                        qmin[i] = min(lo_m, hi_m)
                        qmax[i] = max(lo_m, hi_m)
                    # 6th joint (if present) keeps generous roll
                    if self.dof > 5:
                        if qmin[5] is None: qmin[5] = -math.pi
                        if qmax[5] is None: qmax[5] =  math.pi
                    if any(v is not None for v in qmin[:self.dof]):
                        joint_min = [(-math.pi if qmin[i] is None else qmin[i]) for i in range(self.dof)]
                        joint_max = [( math.pi if qmax[i] is None else qmax[i]) for i in range(self.dof)]
                # Workspace from ee_bounds_m
                ee_bounds = cfg.get('ee_bounds_m')
                if isinstance(ee_bounds, dict):
                    try:
                        mn = list(map(float, ee_bounds.get('min', [])))
                        mx = list(map(float, ee_bounds.get('max', [])))
                        if len(mn) == 3 and len(mx) == 3:
                            ws_min = (mn[0], mn[1], max(0.0, mn[2]))
                            ws_max = (mx[0], mx[1], mx[2])
                    except Exception:
                        pass
                # Optional extras
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
        if self.gripper_first:
            lows = np.array([-1] + [-0.05]*dof, np.float32)
            highs = np.array([ 1] + [ 0.05]*dof, np.float32)
        else:
            lows = np.array([-0.05]*dof + [-1], np.float32)
            highs = np.array([ 0.05]*dof + [ 1], np.float32)
        self.single_action_space = gymnasium.spaces.Box(low=lows, high=highs, dtype=np.float32)

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
        if self.gripper_first:
            # [gripper, dq1..dqN] -> C expects [dq1..dqN, gripper]
            self.actions[:, :self.dof] = actions[:, 1:]
            self.actions[:, self.dof] = actions[:, 0]
        else:
            self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = [binding.vec_log(self.c_envs)]
        return self.observations, self.rewards, self.terminals, self.truncations, info

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


