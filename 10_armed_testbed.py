import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


# Main Code
n_bandit = 2000  # number of bandit problems
n_arms = 10  # number of arms in each bandit problem
n_steps = 1000  # number of time steps

q_true = np.random.normal(loc=0, scale=1, size=(n_bandit, n_arms))  # the true means q*(a) for each arm for all bandits
opt_arms_true = np.argmax(q_true, axis=1)  # the true optimal arms in each bandit

epsilons = [0, 0.01, 0.1]  # epsilon values
colors = sns.color_palette(palette='Set2', n_colors=3)  # colors for plotting each epsilon

sns.set_theme(style='darkgrid')
fig_1, ax_1 = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3.5), layout='constrained')
fig_2, ax_2 = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3.5), layout='constrained')

for epsilon, color in zip(epsilons, colors):
    print('Calculating for epsilon value of : ' + str(epsilon) + '\n...')

    Q = np.zeros((n_bandit, n_arms))  # initialize Q
    N = np.ones((n_bandit, n_arms))  # initialize N
    Qi = np.random.normal(q_true, scale=1)  # initial step of all arms

    R_eps = [0, np.mean(Qi)]
    R_eps_opt = []

    for step in range(2, n_steps + 1):
        R_step = []  # store all rewards for each step
        n_arm_step_opt = 0  # number of the step of best arm
        for i in range(n_bandit):
            if random.random() < epsilon:
                j = np.random.randint(n_arms)  # to explore
            else:
                j = np.argmax(Q[i])  # to exploit
            if j == opt_arms_true[i]:  # to calculate optimal action
                n_arm_step_opt = n_arm_step_opt + 1

            R_temp = np.random.normal(q_true[i][j], scale=1)
            R_step.append(R_temp)  # to store the reward
            N[i][j] = N[i][j] + 1
            Q[i][j] = Q[i][j] + (R_temp - Q[i][j]) / N[i][j]

        R_step_avg = np.mean(R_step)
        R_eps.append(R_step_avg)
        R_eps_opt.append(float(n_arm_step_opt) * 100 / n_bandit)
    ax_1.plot(range(0, n_steps + 1), R_eps, color=color, linewidth=2, label='epsilon=' + str(epsilon))
    ax_2.plot(range(2, n_steps + 1), R_eps_opt, color=color, linewidth=2, label='epsilon=' + str(epsilon))

ax_1.set_title('Average Reward Vs Steps for 10-Armed Testbed')
ax_1.set_ylabel('Average Reward')
ax_1.set_xlabel('Steps')
ax_1.set_ylim(0, 2)
ax_1.legend(loc='best')
plt.savefig('figures/Average Reward Vs Steps for 10-Armed Testbed.png')

ax_2.set_title('Optimal Action Vs Steps for 10-Armed Testbed')
ax_2.set_ylabel(' % Optimal Action')
ax_2.set_xlabel('Steps')
ax_2.set_ylim(0, 100)
ax_2.legend(loc='best')
plt.savefig('figures/Optimal Action Vs Steps for 10-Armed Testbed.png')

plt.show()
