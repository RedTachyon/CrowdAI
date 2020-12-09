import torch
from torch import Tensor
import matplotlib.pyplot as plt
from utils import Timer


def discount_rewards_to_go(rewards: Tensor,
                           dones: Tensor,
                           gamma: float = 1.,
                           ep_len: int = 0) -> Tensor:
    """
    Computes the discounted rewards to go, handling episode endings. Nothing unusual.
    """
    if ep_len:
        rewards = rewards.view(-1, ep_len)

        current_reward = 0
        discounted_rewards = []
        for reward in rewards.T.flip(0):
            current_reward = reward + gamma * current_reward
            discounted_rewards.insert(0, current_reward)
        result = torch.stack(discounted_rewards).T.reshape(-1)
        return result

    else:
        current_reward = 0
        discounted_rewards = []
        for reward, done in zip(rewards.flip(0), dones.flip(0)):
            if done:
                current_reward = 0
            current_reward = reward + gamma * current_reward
            discounted_rewards.insert(0, current_reward)
        return torch.tensor(discounted_rewards)


torch.manual_seed(0)
# rewards = torch.rand(9000)
# dones = torch.tensor([False if (i+1) % 500 else True for i in range(9000)])

rewards = torch.rand(1152000)
dones = torch.tensor([False if (i+1) % 500 else True for i in range(1152000)])


# rewards = torch.tensor([1.] * 5 + [2.] * 5)
# dones = torch.tensor([False if (i+1) % 5 else True for i in range(10)])


t = Timer()
print("Starting discounting")
r_fast = discount_rewards_to_go(rewards, dones, 1., ep_len=500)
print(f"Discounting done in {t.checkpoint()} second")
plt.plot(r_fast[:9000])
plt.title("Fast")
plt.show()

t.checkpoint()
print("Starting slow discounting")
r_simple = discount_rewards_to_go(rewards, dones, 1., ep_len=0)
time = t.checkpoint()
print(f"Slow discounting done in {t.checkpoint()} seconds")
plt.plot(r_simple[:9000])
plt.title("Simple")
plt.show()
