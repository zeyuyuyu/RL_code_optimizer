"""
train.py – REINFORCE + baseline + learned reward mix.

"""
import os, json, numpy as np, torch
from env import CodeOptimizeEnv
from policy import PolicyNet
from functions import functions

# ------------------ 超参 ------------------
cfg = dict(alpha=0.8, beta=0.2, max_steps=10,
           episodes=5000, pretrain=200, gamma=0.95,
           buffer=50, batch=20, lr=5e-4,
           ckpt_interval=100)

# ------------------ 初始化 ------------------
env = CodeOptimizeEnv(functions, cfg)
obs_dim, act_dim = env.reset().shape[0], 6
policy = PolicyNet(obs_dim, act_dim)
opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

hist, batch = [], []
os.makedirs("checkpoints", exist_ok=True)

# ------------------ 训练循环 ------------------
for ep in range(cfg["episodes"]):
    s, logs, ents, rs, done = env.reset(), [], [], [], False
    while not done:
        st = torch.tensor(s).float().unsqueeze(0)
        probs = policy(st)[0]
        a = np.random.choice(act_dim, p=probs.detach().numpy())
        logs.append(torch.log(probs[a] + 1e-8))
        ents.append(-(probs * torch.log(probs + 1e-8)).sum())

        s, r, done, _ = env.step(a)
        rs.append(r)

    G = 0.0
    rets = []
    for r in reversed(rs):
        G = r + cfg["gamma"] * G
        rets.insert(0, G)
    hist.append(sum(rs))
    batch.append((logs, ents, rets))

    #—— 训练 reward-model ————————————————
    if ep >= cfg["pretrain"] and ep % cfg["buffer"] == 0:
        env.rm.fit()

    #—— 更新策略 ————————————————
    if (ep + 1) % cfg["batch"] == 0:
        all_rets = [r for _, _, rets in batch for r in rets]
        baseline = np.mean(all_rets)
        loss = 0.0
        count = 0
        for (logs, ents, rets) in batch:
            for lp, ent, ret in zip(logs, ents, rets):
                adv = ret - baseline
                loss += -lp * adv - 0.01 * ent
                count += 1
        loss /= count
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch.clear()

    #—— 打印 & Checkpoint ————————————————
    if (ep + 1) % 50 == 0:
        print(f"[{ep+1}/{cfg['episodes']}] avg_R={np.mean(hist[-50:]):.2f}")

    if (ep + 1) % cfg["ckpt_interval"] == 0:
        ckpt_path = f"checkpoints/policy_ep{ep+1}.pt"
        torch.save(policy.state_dict(), ckpt_path)
        print(f"checkpoint saved → {ckpt_path}")

# ------------------ 保存曲线 & 最终权重 ------------------
json.dump(hist, open("learning_curve.json", "w"))
torch.save(policy.state_dict(), "policy_final.pt")
print("训练完成：learning_curve.json & policy_final.pt 已保存")
