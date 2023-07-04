import os

import pandas
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
from util import *
from MetasploitENV import MetasploitENV2_0 as metaEnv
import csv
from plotPPO import *

device = torch.device('cpu')


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# PPO agent class
class PPO:
    def __init__(self, environment, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.env = environment
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        PrintExecuting("Update policy")
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save_model(self, path):
        PrintExecuting("Save model")
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def test(self, try_quantity=200):
        privilege_escalation = 0
        gather_hashdump = 0
        lateral_movement = 0
        num_lateral_movement = 0
        PrintTestingPhase()
        state = self.env.reset()
        done = False
        total_reward = 0
        num_of_act = 0
        while not done:
            if num_of_act == try_quantity:
                done = True
                break
            action = self.select_action(state)
            reward, next_state, done = self.env.step(action)
            if reward > 0:
                if self.env.action_list.loc[action, "Type Module"] == "privilege escalation":
                    privilege_escalation = num_of_act + 1
                elif self.env.action_list.loc[action, "Type Module"] == "gather information":
                    gather_hashdump = num_of_act + 1
                else:
                    lateral_movement = num_of_act + 1
                    num_lateral_movement += 1

            PrintInfo(next_state)
            total_reward += reward
            state = next_state
            num_of_act += 1

        print("Test Complete: Total Reward = {}".format(total_reward))
        return privilege_escalation, gather_hashdump, lateral_movement, num_lateral_movement


def train_agent(agent, num_episodes=800, try_quantity=200, save_interval=1, rewards_file="PPO_rewards.csv"):
    PrintTranningPhase()
    rewards = []  # List to store the rewards of each episode
    episode_times = []  # List to store the time taken for each episode

    for episode in range(num_episodes):
        start_time = time.time()  # Start time of the episode
        state = agent.env.reset()
        done = False
        num_of_act = 0
        total_reward = 0
        while not done:
            if num_of_act == try_quantity:
                done = True
                break
            action = agent.select_action(state)
            print(f"----episode {episode + 1} - Step {num_of_act + 1} --------------------")
            reward, state, done = agent.env.step(action)
            PrintInfo(f"Reward: {reward}")
            PrintInfo(f"State: {state}")

            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            num_of_act += 1
            total_reward += reward
        print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))

        episode_time = time.time() - start_time  # Time taken for the episode
        if (episode + 1) % (save_interval * 4) == 0:
            agent.save_model(f"2_PPO_output_model/PPO_model_{num_episodes*2}.pt")

        if (episode + 1) % (save_interval * 2) == 0:
            agent.update()
        if os.path.exists(rewards_file):
            temp = pd.read_csv(rewards_file)
            rewards = temp["Rewards"].tolist()
            episode_times = temp["Episode Time"].tolist()
        # Save rewards and episode times to CSV file
        rewards.append(total_reward)
        episode_times.append(episode_time)
        df = pandas.DataFrame({"Rewards": rewards, "Episode Time": episode_times})
        df.to_csv(rewards_file)


if os.path.exists("2_PPO_output_model") is not True:
    os.mkdir("2_PPO_output_model")
K_epochs = 20

eps_clip = 0.2
gamma = 0.99

# lr_actor = 0.0003
# lr_critic = 0.001

lr_actor = 0.005
lr_critic = 0.003

episode = 5000

if os.path.exists("2_PPO_output_model") is not True:
    os.mkdir("2_PPO_output_model")

env = metaEnv.EnvironmentTraining(package_num=1)
env.setup()
agent = PPO(env, len(env.state), len(env.action_list), lr_actor, lr_critic, gamma, K_epochs, eps_clip)
train_agent(agent, num_episodes=episode)

env = metaEnv.EnvironmentTraining(package_num=2)
env.setup()
agent = PPO(env, len(env.state), len(env.action_list), lr_actor, lr_critic, gamma, K_epochs, eps_clip)
agent.load_model(f"2_PPO_output_model/PPO_model_{episode*2}.pt")
train_agent(agent, num_episodes=episode)

# PlotPPO()
number_of_test = 100
for pkg_num in range(1, 5):
    pe = 0
    n_pe = 0
    gh = 0
    n_gh = 0
    lm = 0
    n_lm = []
    name_test_log = f"Result/Testing/PPO/package_{pkg_num}_{episode*2}.log"
    for num in range(1, number_of_test + 1):
        env = metaEnv.EnvironmentTraining(package_num=1)
        env.setup_test_package(pkg_num)
        env.setup()
        agent = PPO(env, len(env.state), len(env.action_list), lr_actor, lr_critic, gamma, K_epochs, eps_clip)
        agent.load_model(f"2_PPO_output_model/PPO_model_{episode*2}.pt")
        privilege_escalation, gather_hashdump, lateral_movement, num_lateral_movement = agent.test()
        if num_lateral_movement >= env.peers_info.shape[0]:
            lm += 1
        if privilege_escalation > 0:
            n_pe += 1
        if gather_hashdump > 0:
            n_gh += 1
        pe += privilege_escalation
        gh += gather_hashdump
        n_lm.append(num_lateral_movement)
    print("======================================")
    file_write = open(name_test_log, "a")
    file_write.writelines(f'num of privilege escalation success: {n_pe}/100\n')
    file_write.writelines(f'num of gather hashdump success: {n_gh}/100\n')
    file_write.writelines(f"privilege escalation: {pe / n_pe}\n")
    file_write.writelines(f"gather information: {gh / n_gh}\n")
    file_write.writelines(f"lateral movement successful all: {lm}/100\n")
    file_write.writelines(f"lateral movement compromised each time: {n_lm}\n")
    file_write.writelines("\n\n")
