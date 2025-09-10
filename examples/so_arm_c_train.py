import sys
import torch
import pufferlib.vector
from pufferlib import pufferl


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
    # Prevent config parser from seeing arbitrary terminal args
    sys.argv = [sys.argv[0]]

    # Create creator via ocean registry to use our C-native env
    import pufferlib.ocean
    env_creator = pufferlib.ocean.env_creator('puffer_so_arm')
    vecenv = pufferlib.vector.make(
        env_creator,
        num_envs=4,
        num_workers=4,
        batch_size=4,
        backend=pufferlib.vector.Multiprocessing,
        # Important: do NOT also set SoArm(num_envs>1) or agents multiply
        env_kwargs={},
    )

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    policy = Policy(vecenv.driver_env).to(device)

    args = pufferl.load_config('default')
    args['train']['env'] = 'puffer_so_arm'
    args['train']['device'] = device
    # Ensure segments (=batch_size/horizon) >= total_agents (=num_envs)
    args['train']['batch_size'] = 2048
    args['train']['bptt_horizon'] = 128  # 2048/128 = 16 segments
    args['train']['minibatch_size'] = 256
    args['train']['max_minibatch_size'] = 256
    args['train']['optimizer'] = 'adam'
    args['train']['compile'] = False
    args['train']['total_timesteps'] = 2048 * 5

    trainer = pufferl.PuffeRL(args['train'], vecenv, policy)
    for epoch in range(2):
        trainer.evaluate()
        logs = trainer.train()
        print(f"Epoch {epoch}: {logs}")
    trainer.print_dashboard()
    trainer.close()


