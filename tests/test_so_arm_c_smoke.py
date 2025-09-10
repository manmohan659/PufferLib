import numpy as np

def test_so_arm_smoke_import_and_step():
    import pufferlib.ocean.so_arm.so_arm as so
    env = so.SoArm(num_envs=1)
    env.reset()
    assert env.observations.shape[0] == 1
    a = np.zeros_like(env.actions)
    o, r, t, tr, info = env.step(a)
    assert np.isfinite(r).all()
    env.close()


