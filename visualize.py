"""
Learning curves, value function and greedy policy of the Q-learning agent (model.py).

    python visualize.py [--show]

Trains the agent on FrozenLake 4x4 twice:
- deterministic ice (is_slippery=False): the agent must find the shortest safe path;
- slippery ice (is_slippery=True): each move only goes where intended 1/3 of the time,
  so the agent has to learn a cautious policy.
Saves figures/learning_curves.png and figures/policy_maps.png.
"""
import sys

import gymnasium as gym
import matplotlib

SHOW = "--show" in sys.argv
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from model import evaluate_success_rate, extract_greedy_policy, train_q_learning  # noqa: E402

GAMMA = 0.99
SETTINGS = {
    "Deterministic ice": dict(is_slippery=False, num_episodes=2000, alpha=0.1, epsilon_decay=0.995),
    "Slippery ice": dict(is_slippery=True, num_episodes=20000, alpha=0.05, epsilon_decay=0.9995),
}
ARROWS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # LEFT, DOWN, RIGHT, UP as (dx, dy) in grid coords


def optimal_success_probability(env, horizon=100):
    """Best achievable probability of reaching the goal within `horizon` steps (finite-horizon
    value iteration on the known transition model, used only to check the learned agent)."""
    P = env.unwrapped.P
    n = env.observation_space.n
    V = np.zeros(n)
    for _ in range(horizon):
        V = np.array([max(sum(p * (r + (0.0 if done else 1.0) * V[s2]) for p, s2, r, done in P[s][a])
                          for a in P[s]) for s in range(n)])
    return V[0]


def rolling(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def run(name, cfg):
    env = gym.make("FrozenLake-v1", is_slippery=cfg["is_slippery"])
    q, returns = train_q_learning(env, num_episodes=cfg["num_episodes"], alpha=cfg["alpha"], gamma=GAMMA,
                                  epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=cfg["epsilon_decay"],
                                  seed=0, max_steps=200)
    policy = extract_greedy_policy(q)
    success = evaluate_success_rate(env, policy, num_episodes=1000, seed=0, max_steps=200)
    desc = env.unwrapped.desc.astype(str)
    best = optimal_success_probability(env)
    env.close()
    print(f"{name}: greedy success rate over 1000 episodes = {success:.1%} "
          f"(optimal policy: {best:.1%}), V(start) = {q[0].max():.4f}")
    return dict(q=q, returns=np.array(returns), policy=policy, success=success, best=best, desc=desc, cfg=cfg)


def draw_policy(ax, res, title):
    q, policy, desc = res["q"], res["policy"], res["desc"]
    n = desc.shape[0]
    V = q.max(axis=1).reshape(n, n)
    im = ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
    for s in range(n * n):
        r, c = divmod(s, n)
        cell = desc[r, c]
        if cell == "H":
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#3b3b3b"))
            ax.text(c, r, "hole", ha="center", va="center", color="white", fontsize=8)
            continue
        if cell == "G":
            ax.text(c, r, "goal", ha="center", va="center", fontsize=9, fontweight="bold", color="#b30000")
            continue
        dx, dy = ARROWS[int(policy[s])]
        ax.annotate("", xy=(c + 0.32 * dx, r + 0.32 * dy), xytext=(c - 0.32 * dx, r - 0.32 * dy),
                    arrowprops=dict(arrowstyle="-|>", color="#b30000", lw=1.6))
        ax.text(c, r + 0.38, f"{V[r, c]:.2f}", ha="center", va="center", fontsize=7)
        if cell == "S":
            ax.text(c - 0.36, r - 0.36, "S", fontsize=8, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\ngreedy success {res['success']:.1%} (optimal: {res['best']:.1%})", fontsize=10)
    return im


if __name__ == "__main__":
    results = {name: run(name, cfg) for name, cfg in SETTINGS.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (name, res) in zip(axes, results.items()):
        cfg = res["cfg"]
        w = max(20, cfg["num_episodes"] // 50)
        ax.plot(np.arange(w - 1, len(res["returns"])), rolling(res["returns"], w), color="#2171b5", label=f"training success rate ({w}-episode mean, with exploration)")
        eps = np.maximum(0.05, cfg["epsilon_decay"] ** np.arange(cfg["num_episodes"]))
        ax.plot(eps, color="#d94801", ls="--", lw=1, label=r"exploration rate $\varepsilon$")
        ax.set_ylim(0, 1.02); ax.set_xlabel("Training episode"); ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("figures/learning_curves.png", dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for ax, (name, res) in zip(axes, results.items()):
        im = draw_policy(ax, res, name)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.03, label=r"$V(s)=\max_a Q(s,a)$")
    fig.savefig("figures/policy_maps.png", dpi=140, bbox_inches="tight")
    print("saved figures/learning_curves.png and figures/policy_maps.png")
    if SHOW:
        plt.show()
