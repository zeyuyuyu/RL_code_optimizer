"""
train.py
Training script for RL code optimizer (REINFORCE + baseline).
"""
import torch, numpy as np, json
from env import CodeOptimizeEnv
from policy import PolicyNet
from functions import functions

env = CodeOptimizeEnv(functions)
obs_dim = env.reset().shape[0]
act_dim = len(env.action_space)

policy = PolicyNet(obs_dim, act_dim)
optim  = torch.optim.Adam(policy.parameters(), lr=1e-2)

history, batch = [], []
episodes = 500

for ep in range(episodes):
    state = env.reset()
    logps, rewards, done = [], [], False
    while not done:
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(st)[0].detach().numpy()
        a = np.random.choice(act_dim, p=probs)
        logp = torch.log(policy(st)[0, a] + 1e-8)
        state, r, done, _ = env.step(a)
        logps.append(logp); rewards.append(r)
    G = sum(rewards); history.append(G); batch.append((logps, G))

    if (ep + 1) % 10 == 0:
        baseline = np.mean([g for _, g in batch])
        loss = sum((-lp * (ret - baseline) for logs, ret in batch for lp in logs)) / len(batch)
        optim.zero_grad(); loss.backward(); optim.step(); batch.clear()
        print(f"Episode {ep+1}, avg return {baseline:.2f}")

with open("learning_curve.json", "w") as f:
    json.dump(history, f)
print("Done. Saved learning_curve.json")
