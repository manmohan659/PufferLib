import os
import sys
import torch
import functools
import pufferlib.vector
from pufferlib import pufferl

from pufferlib.environments.so_arm_bullet import env_creator
from pufferlib.environments.so_arm_bullet.environment import single_env_creator


class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(),
        )
        self.mu_head = torch.nn.Linear(256, act_dim)
        # Global log std parameter for continuous actions
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))
        self.value_head = torch.nn.Linear(256, 1)

    def forward_eval(self, obs, state=None):
        x = self.net(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, self.value_head(x)

    def forward(self, obs, state=None):
        return self.forward_eval(obs, state)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/so_arm_bullet_train.py <URDF_PATH>")
        sys.exit(1)
    urdf_path = sys.argv[1]

    # Prevent pufferl.load_config from interpreting the URDF path as a CLI arg
    sys.argv = [sys.argv[0]]

    # Visible training: force GUI=True via single_env_creator
    creator = functools.partial(
        single_env_creator,
        urdf_path=urdf_path,
        capture_video=False,
        gamma=0.99,
        pufferl=True,
        gui=False,
    )
    # For visible training and simpler debugging, use a single Serial env
    vecenv = pufferlib.vector.make(
        creator,
        num_envs=1,
        num_workers=1,
        batch_size=1,
        backend=pufferlib.vector.Serial,
    )

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    policy = Policy(vecenv.driver_env).to(device)

    args = pufferl.load_config('default')
    args['train']['env'] = 'so_arm_bullet'
    args['train']['device'] = device
    # Keep minibatch_size <= batch_size to satisfy trainer assertions
    args['train']['batch_size'] = 1024
    args['train']['bptt_horizon'] = 256
    args['train']['minibatch_size'] = 256
    args['train']['max_minibatch_size'] = 256
    args['train']['optimizer'] = 'adam'  # avoid heavyball on MPS
    args['train']['compile'] = False
    args['train']['total_timesteps'] = 1024 * 10

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

    for epoch in range(3):
        trainer.evaluate()
        logs = trainer.train()
        print(f"Epoch {epoch}: {logs}")

    trainer.print_dashboard()
    trainer.close()

