import pandas as pd
import matplotlib.pyplot as plt


def PlotA2C():
    df = pd.read_csv("A2C_rewards.csv")
    df = df.rename(columns={"Unnamed: 0": "Episode"})
    df['Rewards'] = df['Rewards'].rolling(window=50, win_type='triang', min_periods=1).mean()
    x = df['Episode']
    y = df['Rewards']
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linestyle='solid', linewidth=2, alpha=1)
    plt.grid(True)
    plt.title(f"A2C Training {len(x)} episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"Result/Training/A2C_{len(x)}.png")

#
# PlotA2C()
