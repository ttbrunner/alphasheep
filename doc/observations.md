Writing down random ideas and observations.

---
Commit #5c34e5723726bb9c62c6a7ffb92ce101e13ba409
- DQNAgent, playing Herz Solo exclusively, while only seeing its own cards and the current trick
- RandomCardAgent achieves 60.4% win rate (mean over 100k episodes).

Observation:
- DQNAgent, in its current state, has 60.5% (10k episodes), so learning doesn't seem to work for now (or too slow).
- Surely, some tuning is needed.

Observation:
- If we change the activation from 'linear' to 'softmax', suddenly it learns much better! 65% win rate (10k episodes)!
- Why? The Q-values should certainly be linear.
- So softmax clearly has some numerical effect that accelerates learning in our setup
    - Values are squashed to 1/0 (like sigmoid)
    - Values influence each other (unlike sigmoid/linear)
- Idea: what happens if we use sigmoid instead of softmax?
- Goal: Revert to linear (because this is the "correct" q-learning paradigm), but tune something else to get the same good learning.

---
Commit #7c0a5247860eafd075458055add9e472c8afc61f
- Not doing any of that softmax stuff, we're sticking to the recipe for now.
- Changed training to not train after *every single* experience, but only every 8 experiences. The idea being that there is 0 reward for most plays, and the reward signal comes only every 8 plays. Therefore we waste a lot of time training on little information. Now we train only when there are 8 new experiences in the buffer. 
- **Note**: Conceptually, this might have broken the dual network approach (sync_models), as they are always synced at every training step now.
- In any case, learning seems to work now. The agent sometimes achieves >70% winrate (SMA over last 1000 episodes).
- Also, training is *much* faster!
- Need performance measures for meaningful evaluation! 



---
Commit #d916b78de686f45c962518eec22b3b86252efab4
- DQNAgent is still playing Herz Solo exclusively (as the declaring player).
- New performance measure for evaluation: relative winrate as compared to a baseline (RandomCardAgent)
- Typically trains to ~1.20. We have a stray 1.28 but this was lucky.
- Training oscillates between 1.00-1.20 a lot. Not unexpected, we can see something similar if we run tutorials with the inverse pendulum!
- We notice some emerging behaviors:
    - DQNAgent will never "schmier" if they can instead take a trick (good)    
    - Often though, it plays cards that will surely lose the trick (bad)
    - **Interpretation**: the agent learned to prevent some obvious mistakes, but otherwise doesn't seem to do anything smart. 
     
    

---
Commit #f2fd24a91222346837e0af232804f744658c32f3
- DQNAgent: extended state with info about all cards that have been played (=knowledge: the enemy cannot have them).
- Theoretically, this allows the agent to play better if they know how to use the information.

Observation:
- We would have expected this to improve performance, but so far it didn't do anything. 
- In fact, it seems that training is now a bit harder. Best checkpoints are now at ~1.20, but almost never above. We had lucky 1.22+ before that.
- **Idea**: Is the way we represent state bad? We have huge one-hot vectors. 

Observation:
- Watching the agent play, in the first 3-5 tricks, the Q-vector is exactly the same - regardless of state.
- Trumps are near the top, so the agent still plays somewhat competently in that stage, but really it doesn't react to the state at all.
- Around the 4th trick, the agent suddenly "wakes up" and Q-values start changing. From that point on it seems to make pretty good moves.
- **Interpretation**: Learning seems to work close to the end of the game, but not earlier. For anything that is further away, a "constant" policy is adopted that mostly works. Why? 
- **Idea**: Q-values are rather small in the earlier tricks. Maybe that adversely affects learning?
- **Idea**: Could changing the discount factor help?
- **Idea**: Could a larger network help? Maybe it is underfitting, so only the most important connections are drawn.
- **Idea**: Currently, we set unavailable actions to 0 reward, but never allow the agent to execute them. 
    - This allows the agent to win many situations with exactly the same Q-vector - invalid actions are automatically filtered. 
    - If we instead allowed the agent to play those actions, deliver -1 reward and repeat the same state, it will be forced to change the q-values for invalid actions on every turn.
    - Will that "fix" this static behavior?
    - Will it make training harder?
    - Will it ultimately increase performance?
    
---
Commit #df3744fc4019241b30bc33fdc402692d17ff62a8

Created RuleBasedAgent.
- This agent has hard-coded if-else behavior that mirror the same rules that are taught to human players. 
- Currently, it can play a solo both as declaring player and as non-declaring player (in the opposite team). It plays roughly like a human beginner would.
- For now, the rules are very simple, and it does NOT have any memory of previously played cards whatsoever.
- This means that it plays with less information than DQNAgent, so it should be possible for DQNAgent at least match its performance!

Observation:
- RuleBasedPlayer achieves a rel. performance of 1.38 (compared to RandomCardAgent) => much better than DQNAgent!
- **Interpretation**: 1.38 can serve as the new benchmark for achieving super-rulebased performance.

Next steps:
- **Idea**: Start tuning hyperparams for DQNAgent and implement some advanced methods (double DQN?). How close can we get to 1.38?
- **Idea**: Train against RandomCardAgent instead of RuleBasedAgent without changing anything. Will the performance improve?
- **Idea**: Resolve the "constant q-vector" problem above. Does this help?

---
Commit #a0a6c537c5e30d7c27c1d7f0e52e637b828f3f7e

Increased network size from (256, 128, 64) to (384, 256, 128 neurons). This roughly triples the number of parameters (43008 -> 135168).
Observation: 
- Much better! Now DQNAgent achieves a rel. performance of 1.30 (best so far).
- Also, performance was much stronger from the start (above 1.23 in the first hour of training alone). 

Observation:
- In an early checkpoint (1.24), the "constant q-vector" behavior observed only in the first 1-3 tricks (before: 3-5).
- In a later checkpoint (1.30), this behavior only occurs very sporadically. 
- **Interpretation**: Overparametrizing the network greatly helps (in this case).
- **Interpretation**: Better performance correlates with less "constant q-vectors".
- **Idea**: Is it really overparametrized? How far can we go? How large is too large?

---
Commit #12480153b371c7d64721b37c23c7b98fc20435d7

Now training against RuleBasedAgent. However, performance is still evaluated against RandomCardAgent.

Observation:
- Worse instead of better. Agent doesn't improve beyond 1.20.
- Apparently, it learns a completely static policy, with most trumps ranked at the top.
- The "constant q-vector" problem is back with a vengeance. 
- **Interpretation**: This seems to happen whenever the agent couldn't learn a good way to react to a situation / doesn't know what to do.

Discussion:
- Currently, static policies are just *too* good. Since invalid actions are automatically discarded, a static policy actually generates fitting actions for various situations. Not necessarily the best ones, but often kind of OK.
- It seems the agent learned a static policy that maximizes the expected future reward, and can actually achieve a respectable performance with it (1.20)!
- **Idea**: Extract the policy (q-vector), create a StaticPolicyAgent, and evaluate it. Is its performance similarly high?

Next steps:
- Verify our assumption with a StaticPolicyAgent
- Change evaluation to also evaluate against RuleBasedAgent (this will yield new numbers)
- Introduce infinite negative reward for invalid actions (or a feedback loop of -1): this should force the agent away from static policies.
