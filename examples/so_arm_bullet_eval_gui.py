import sys
import time
import numpy as np

import gymnasium

from pufferlib.environments.so_arm_bullet.environment import SOArmBulletGym


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/so_arm_bullet_eval_gui.py <URDF_PATH>")
        sys.exit(1)
    urdf_path = sys.argv[1]

    env = SOArmBulletGym(urdf_path=urdf_path, gui=True)
    obs, info = env.reset()
    print("Reset. obs shape:", obs.shape)

    for t in range(500):
        # Simple scripted behavior: small random dq, attempt to close gripper near the object
        ee = obs[env.dof : env.dof + 3]
        obj = obs[env.dof + 3 : env.dof + 6]
        grip = 1.0 if np.linalg.norm(ee - obj) < 0.05 else -1.0
        action = np.concatenate([np.random.uniform(-0.01, 0.01, size=env.dof), [grip]]).astype(np.float32)
        obs, rew, term, trunc, info = env.step(action)
        if t % 50 == 0:
            print(f"t={t} rew={rew:.3f} dist={info['dist_to_goal']:.3f} grasped={info['grasped']}")
        time.sleep(1.0 / 60.0)

    env.close()

