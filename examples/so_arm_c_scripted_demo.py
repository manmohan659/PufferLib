import time
import numpy as np

from pufferlib.ocean.so_arm.so_arm import SoArm


def A(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,      ca,        d],
        [0.0,    0.0,     0.0,      1.0],
    ], dtype=np.float32)


def fk_pos(q, a, alpha, d):
    T = np.eye(4, dtype=np.float32)
    for i in range(len(q)):
        T = T @ A(a[i], alpha[i], d[i], q[i])
    return T[:3, 3].copy()


def jacobian_pos(q, a, alpha, d, eps=1e-4):
    J = np.zeros((3, len(q)), dtype=np.float32)
    p0 = fk_pos(q, a, alpha, d)
    for i in range(len(q)):
        dq = q.copy(); dq[i] += eps
        p1 = fk_pos(dq, a, alpha, d)
        J[:, i] = (p1 - p0) / eps
    return J


def dls_step(q, p_goal, a, alpha, d, lam=1e-2, max_step=0.05):
    p = fk_pos(q, a, alpha, d)
    e = (p_goal - p).reshape(3, 1)
    J = jacobian_pos(q, a, alpha, d)
    JJt = J @ J.T
    dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(3), e)
    dq = dq.squeeze()
    # clamp
    infnorm = np.max(np.abs(dq))
    if infnorm > max_step:
        dq *= max_step / (infnorm + 1e-8)
    return dq.astype(np.float32)


def main():
    # Use full 6-DoF (includes wrist roll as joint 6). Pass servo and gripper params to showcase features.
    DOF = 6
    env = SoArm(
        num_envs=1,
        dof=DOF,
        ws_min=(-0.2, -0.2, 0.0),
        ws_max=(0.4, 0.4, 0.4),
        grasp_radius=0.025,
        # Simple first-order lag (steps) and per-step rate limits (rad/step)
        servo_tau=[0.0, 1.5, 1.2, 1.0, 0.8, 0.5][:DOF],
        joint_rate=[0.05, 0.04, 0.04, 0.05, 0.05, 0.08][:DOF],
        # Parallel-jaw gripper opening (meters)
        jaw_min=0.0,
        jaw_max=0.04,
        grip_speed=0.004,
        seed=42,
    )
    obs, _ = env.reset()

    # Pull DH from wrapper defaults (kept in constructor args we passed)
    # We don't have them on the Python object, so mirror the default values here
    a = np.array([0.0304, 0.1160, 0.1347, 0.0, 0.0, 0.0], dtype=np.float32)
    alpha = np.array([np.pi/2, 0.0, 0.0, -np.pi/2, 0.0, 0.0], dtype=np.float32)
    d = np.array([0.0542, 0.0, 0.0, 0.0, 0.0609, 0.0], dtype=np.float32)

    stage = 0  # 0: go to red ball (obj), 1: close, 2: go to bin (goal center), 3: open
    t0 = time.time()
    # Run for ~90 seconds at ~60 FPS (or at least 1000 steps)
    for t in range(5400):
        # Unpack current state
        dof = DOF
        q = obs[0, :dof].astype(np.float32)
        ee = obs[0, dof:dof+3]
        obj = obs[0, dof+3:dof+6]
        goal = obs[0, dof+6:dof+9]
        grasped = bool(obs[0, dof+9] > 0.5)

        # State machine
        if stage == 0:
            target = obj
            if np.linalg.norm(ee - obj) < 0.03:
                stage = 1
        elif stage == 1:
            target = obj
            if not grasped:
                action = np.zeros((1, dof + 1), dtype=np.float32)
                action[0, -1] = 1.0  # close (gripper velocity)
                obs, r, dN, trN, info = env.step(action)
                env.render(); continue
            else:
                stage = 2
                target = goal
        elif stage == 2:
            target = goal  # move ball to bin center
            if np.linalg.norm(obj - goal) < 0.03:
                stage = 3
        else:
            target = goal
            # open
            action = np.zeros((1, dof + 1), dtype=np.float32)
            action[0, -1] = -1.0  # open
            obs, r, dN, trN, info = env.step(action)
            env.render(); continue

        # IK step toward target
        dq = dls_step(q, target.astype(np.float32), a, alpha, d, lam=1e-2, max_step=0.05)
        action = np.zeros((1, dof + 1), dtype=np.float32)
        action[0, :dof] = dq
        # Slightly close jaws when approaching object; fully close when transporting
        action[0, -1] = 0.3 if stage < 2 else 1.0
        obs, r, dN, trN, info = env.step(action)
        env.render()
        # Try to hold ~60 FPS
        time.sleep(1/60)

    env.close()


if __name__ == "__main__":
    main()


