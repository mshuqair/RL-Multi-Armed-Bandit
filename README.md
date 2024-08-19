
# Multi-Armed Bandit
- Implementation of Multi-Armed Bandits uing epsilon-greedy algorithm in Python.
- This implementation is a modified version of: https://github.com/SahanaRamnath/MultiArmedBandit_RL
- This implementation employs the algorithm for Test bed and Ads Optimization Dataset
- The rest of the documentation is coming soon.

## Introduction
Multi-Armed Bandit (MAB) is a Machine Learning framework in which an agent has to select actions (also called arms) in order to maximize its cumulative reward in the long term. In each round, the agent receives some information about the current state (context), then it chooses an action based on this information and the experience gathered in previous rounds. At the end of each round, the agent receives the reward associated with the chosen action.
In a simple form, the Multi-Armed Bandit problem is as follows: you are faced with k slot machines (i.e., an “k -armed bandit”). When the arm on a machine is pulled, it has some unknown probability of dispensing a unit of reward (e.g., $1). The task is to pull one arm at a time so as to maximize the total rewards accumulated over time (the best payout, while not losing too much money).
Trying each machine once and then choosing the one that paid the most would not be a good strategy: The agent could fall into choosing a machine that had a lucky outcome in the beginning but is suboptimal in general. Instead, the agent should repeatedly come back to choosing machines that do not look so good, in order to collect more information about them. This is the main challenge in Multi-Armed Bandits: the agent has to find the right mixture between exploiting prior knowledge and exploring so as to avoid overlooking the optimal actions.


## Definition


## The 10-Armed Test Bed


## Ads Dataset


## Conclusions



## References


## Code Requirements
The code was run and tested using the following:
- Python		3.10.11
- matplotlib	3.9.0
- seaborn		0.13.2
- numpy			1.26.3
