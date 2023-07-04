import pandas as pd
import matplotlib.pyplot as plt


def PlotPPO():
    df = pd.read_csv("PPO_rewards.csv")
    df = df.rename(columns={"Unnamed: 0": "Episode"})
    df['Rewards'] = df['Rewards'].rolling(window=50, win_type='triang', min_periods=1).mean()
    plt.figure(figsize=(10, 6))
    x = df['Episode']
    y = df['Rewards']
    plt.plot(x, y, linestyle='solid')
    plt.grid(True)
    plt.title(f"PPO Training {len(x)} episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f"Result/Training/PPO_{len(x)}.png")