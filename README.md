# MONTE CARLO CONTROL ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

### States
#### 5 Terminal States:
  G (Goal): The state the agent aims to reach.

  H (Hole): A hazardous state that the agent must avoid at all costs.
#### 11 Non-terminal States:
  S (Starting state): The initial position of the agent.

  Intermediate states: Grid cells forming a layout that the agent must traverse.
### Actions
   The agent has 4 possible actions:

0: Left

1: Down

2: Right

3: Up
### Transition Probabilities
Slippery surface with a 33.3% chance of moving as intended and a 66.6% chance of moving in orthogonal directions. For example, if the agent intends to move left, there is a

33.3% chance of moving left, a
33.3% chance of moving down, and a
33.3% chance of moving up.

### Rewards
The agent receives a reward of 1 for reaching the goal state, and a reward of 0 otherwise.

### Graphical Representation
![monte-carlo-control](1.png)

## MONTE CARLO CONTROL ALGORITHM
1. Initialize the state value function V(s) and the policy π(s) arbitrarily.

2. Generate an episode using π(s) and store the state, action, and reward sequence.

3. For each state s appearing in the episode:
      
      G ← return following the first occurrence of s

      Append G to Returns(s)

      V(s) ← average(Returns(s))

4. For each state s in the episode:

      π(s) ← argmax_a ∑_s' P(s'|s,a)V(s')

5. Repeat steps 2-4 until the policy converges.
6. Use the function decay_schedule to decay the value of epsilon and alpha.
7. Use the function gen_traj to generate a trajectory.
8. Use the function tqdm to display the progress bar.
9. After the policy converges, use the function np.argmax to find the optimal policy. The function takes the following arguments:

    Q: The Q-table.

    axis: The axis along which to find the maximum value.


## MONTE CARLO CONTROL FUNCTION
Include the Monte Carlo control function

## OUTPUT:
### Name: Pooja A
### Register Number: 212222240072

Mention the Action value function, optimal value function, optimal policy, and success rate for the optimal policy.

## RESULT:

We have successfully developed a Python program to find the optimal policy for the given RL environment using the Monte Carlo algorithm.
