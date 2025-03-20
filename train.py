import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 环境参数
TARGET_RANGE = (4.0, 6.0)
INITIAL_HEIGHT = 10.0
X_BOUNDS = (0.0, 10.0)


class DropBlockEnv:
    def __init__(self):
        self.x = None
        self.y = None
        self.reset()

    def reset(self):
        self.y = INITIAL_HEIGHT
        self.x = np.random.uniform(X_BOUNDS[0], X_BOUNDS[1])
        # print(self.x, self.y)
        return self._get_state()

    def _get_state(self):
        # 归一化状态到[0,1]范围
        return np.array([self.x / X_BOUNDS[1], self.y / INITIAL_HEIGHT], dtype=np.float32)

    def step(self, action):
        # print(action): 0
        # if action == 0:
        #     print("to the left")
        # else:
        #     print("to the right")

        # 执行动作并更新位置
        new_x = self.x + (1.0 if action else -1.0)
        new_x = np.clip(new_x, X_BOUNDS[0], X_BOUNDS[1])  # 限制移动范围
        # print(new_x): 3.0669907800947014

        self.y -= 1.0
        self.x = new_x
        # print(self.x, self.y): 3.0669907800947014 9.0

        done = self.y <= 0
        reward = self._calculate_reward(done)
        # print(done): False
        # print(reward): -0.29665046099526493

        # print(self._get_state(), reward, done, {}): [0.30669907 0.9       ] -0.29665046099526493 False {}
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, done):
        # print(done): False

        """精细化的奖励计算"""
        if done:
            # 最终奖励
            if TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]:
                return 10.0
            return -10.0

        # 过程奖励
        distance = abs(self.x - np.mean(TARGET_RANGE))
        # print(distance): 1.9330092199052986
        in_target_air = TARGET_RANGE[0] <= self.x <= TARGET_RANGE[1]
        # print(in_target_air): False

        reward = 0.0
        reward += 0.2 if in_target_air else -0.1  # 位置奖励
        reward -= 0.05 * distance  # 距离惩罚
        reward -= 0.1  # 时间惩罚
        # print(reward): -0.29665046099526493

        return reward


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


class PGAgent:
    def __init__(self, state_size=2, gamma=0.99, lr=0.001):
        self.gamma = gamma    # 0.99
        self.net = PolicyNetwork(state_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)    # 0.001
        self.rewards = []
        self.log_probs = []

    def get_action(self, state):
        # print(state): [0.45596555, 1.0]

        state = torch.FloatTensor(state)
        # print(state): tensor([0.4560, 1.0000])
        probs = self.net(state)
        # print(probs): tensor([0.5218, 0.4782], grad_fn=<SoftmaxBackward0>)
        dist = torch.distributions.Categorical(probs)    # 创建一个分类分布（Categorical Distribution）对象
        # print(dist): Categorical(probs: torch.Size([2]))
        action = dist.sample()    # 从分布中采样一个类别
        # print(action): tensor(0)
        self.log_probs.append(dist.log_prob(action))    # 计算采样类别的对数概率
        # print(dist.log_prob(action)): tensor(-0.6505, grad_fn=<SqueezeBackward1>)

        return action.item()

    def update_policy(self):
        # print(self.rewards):
        # [0.08916545413916871, 0.06083454586083131, 0.08916545413916871, 0.06083454586083131, 0.08916545413916871,
        # -0.2608345458608313, 0.08916545413916871, -0.2608345458608313, -0.3108345458608313, -10.0]

        # 计算折扣回报
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        # [-9.447749402674491, -9.633247330114807, -9.792001894924889, -9.980977120266724, -10.143244107199552,
        # -10.335767233675474, -10.176699684661255, -10.369560746263055, -10.210834545860832, -10.0]

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 归一化

        # print(returns):
        # tensor([ 1.8506,  1.2390,  0.7155,  0.0924, -0.4426, -1.0774, -0.5529, -1.1888, -0.6655,  0.0297])

        # print(self.log_probs)

        # 计算损失
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        # print(policy_loss):
        # [tensor(1.3022), tensor(0.8649), tensor(0.4960), tensor(0.0639), tensor(-0.3090),
        #  tensor(-0.7571), tensor(-0.3765), tensor(-0.8428), tensor(-0.4737), tensor(0.0212)]

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        # print(policy_loss): tensor(-0.0109, dtype=torch.float64, grad_fn=<SumBackward0>)

        policy_loss.backward()
        self.optimizer.step()

        # 清空缓存
        self.rewards = []
        self.log_probs = []


# 训练参数
EPISODES = 20000
PRINT_INTERVAL = 100
success_history = deque(maxlen=PRINT_INTERVAL)

# 初始化环境和智能体
env = DropBlockEnv()
agent = PGAgent()

# 初始测试
test_episodes = 10000
success_count = 0
for _ in range(test_episodes):
    state = env.reset()
    reward = None
    # print(state): [0.02351148, 1.0]

    done = False
    while not done:
        action = agent.get_action(state)    # 输入状态state，通过神经网络输出动作action
        # print(action): 0
        state, reward, done, _ = env.step(action)

    if reward > 0:
        success_count += 1

print(f"\nFinal Success Rate: {success_count / test_episodes * 100:.1f}%")


for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    reward = None

    while not done:
        action = agent.get_action(state)
        # print(action): 0
        next_state, reward, done, _ = env.step(action)
        # print(next_state, reward, done): [0.02699641 0.9       ] -0.43650179562709357 False

        agent.rewards.append(reward)
        state = next_state
        total_reward += reward
    # print(agent.rewards):
    # [-0.2596870367615087, -0.30968703676150866, -0.3596870367615087, -0.40968703676150864, -0.44999999999999996,
    # -0.44999999999999996, -0.4, -0.44999999999999996, -0.44999999999999996, -10.0]
    # print(total_reward): -13.538748147046036

    # 记录成功率
    if reward > 0:
        success_history.append(1)
    else:
        success_history.append(0)

    # 更新策略
    agent.update_policy()

    # 输出训练信息
    if episode % PRINT_INTERVAL == 0:
        success_rate = np.mean(success_history) * 100
        print(f"Episode: {episode:4d} | Total Reward: {total_reward:6.1f} | "
              f"Success Rate: {success_rate:5.1f}%")

# 最终测试
test_episodes = 10000
success_count = 0
for _ in range(test_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
    if reward > 0:
        success_count += 1

print(f"\nFinal Success Rate: {success_count / test_episodes * 100:.1f}%")

# 保存网络权重
torch.save(agent.net.state_dict(), 'policy_network_weights.pth')
print("Policy network weights saved to 'policy_network_weights.pth'")