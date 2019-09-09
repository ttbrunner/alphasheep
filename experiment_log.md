## Random ideas and observations.


---
Commit #56b165280c9993a6df2bbda12ee0b74b8a234a2e
- Simulator and GUI are implemented. Here is a first version of DQNAgent.
- DQNAgent plays Herz Solo exclusively against RandomCardAgents, while only seeing its own cards and the current trick.
- We compare its winrate against that of another RandomCardAgent player.

Observation:
- RandomCardAgent achieves 60.4% win rate (mean over 100k episodes).
- DQNAgent, in its current state, has 60.5% (10k episodes), so learning doesn't seem to work for now (or too slow).
- Surely, some tuning is needed.

Observation:
- I know this sounds stupid, but if we change the activation from 'linear' to 'softmax', suddenly it learns much better! 65% win rate (10k episodes)!
- Why? The Q-values should certainly be linear.
- So softmax clearly has some numerical effect that accelerates learning in our setup
    - Values are squashed to 1/0 (like sigmoid)
    - Values influence each other (unlike sigmoid/linear)
- Idea: what happens if we use sigmoid instead of softmax?
- Goal: Revert to linear (because this is the "correct" q-learning paradigm), but tune something else to get the same good learning.

---
Commit #d56dcc90eceaf1cc5baf3c47eb2dae9f4a12f269
- Not doing any of that softmax stuff, we're sticking to the recipe for now.
- Changed training to not train after *every single* experience, but only every 8 experiences. The idea being that there is 0 reward for most plays, and the reward signal comes only every 8 plays. Therefore we waste a lot of time training on little information. Now we train only when there are 8 new experiences in the buffer. 
- **Note**: Conceptually, this might have broken the dual network approach (sync_models), as they are always synced at every training step now.
- In any case, learning seems to work now. The agent sometimes achieves >70% winrate (SMA over last 1000 episodes).
- Also, training is *much* faster!
- Need performance measures for meaningful evaluation! 



---
Commit #b349e28d92c91af206194dc374df87dde0b32bc9
- DQNAgent is still playing Herz Solo exclusively (as the declaring player).
- New performance measure for evaluation: relative winrate as compared to a baseline (RandomCardAgent)
- Typically trains to ~1.20. We have a stray 1.28 but this was lucky.
- Training oscillates between 1.00-1.20 a lot. Not unexpected, we can see something similar if we run tutorials with the inverse pendulum!
- We notice some emerging behaviors:
    - DQNAgent will never "schmier" if they can instead take a trick (good)    
    - Often though, it plays cards that will surely lose the trick (bad)
    - **Interpretation**: the agent learned to prevent some obvious mistakes, but otherwise doesn't seem to do anything smart. 
     
    

---
Commit #3beb70eb9f755f2194a7e5b5ccfc4f56c96b7f4c
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
Commit #2304406836375ccd9cc7a2b26cefe34aa8a05433

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
- **Idea**: Train against RuleBasedAgent instead of RandomCardAgent without changing anything. Will the performance improve?
- **Idea**: Resolve the "constant q-vector" problem above. Does this help?

---
Commit #03d2e79fea633cb6a8ffc2bc5886a76a6f1c84b2

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
Commit #68331d1c5a2635d907a324fc0c59b6802b6013e1

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

---
Commit #695a5254fa47f54eec67732c1d5b88dfc9414f77

Created StaticPolicyAgent with a fixed Q-vector extracted from the previous DQNAgent checkpoint.

Observation:
- It performs just as well as the previous DQNAgent (1.20)!!!
- This confirms our earlier assumption.
- Static policies seem to act like a local minimum, and training gets stuck.

Discussion:
- Playing against stronger enemies causes the agent to prefer this kind of local minimum. 
- Presumably, the task is *too hard* for the agent to learn.
- **Idea**: Is this a problem of exploration? Against the RandomPolicyAgent, it was often possible to win the game with sub-optimal choices and then evolve from there. Against the RuleBasedAgent, we need to play really well in order to win. This means that often there exists only a single sequence of actions that results in a positive reward signal. 
- In our current setup (eps-greedy), the possibility of exploring exactly that sequence is very low.
- Note: The term "optimal" is used loosely here; in fact we cannot determine truly optimal actions because of partial observability.
- **Idea**: Weighted replay buffer? Assign more importance to some experiences?
- **Idea**: Reward shaping? Give rewards per trick, based on points scored? Don't want to do this right now, seems like opening a whole can of worms.
- **Idea**: Curriculum learning? Start against RandomCard, then switch to RuleBased enemies? Perhaps even self-play later?

Next steps:
- Change evaluation to also evaluate against RuleBasedAgent (this will yield new numbers, and select different "best-performing" checkpoints)
- Try infinite negative reward for invalid actions (or a feedback loop of -1): this should force the agent away from static policies. Does this kick-start learning, or will it just give us a different form of bad performance?
- Try reward shaping.
- Try curriculum learning.
- Try some modifications to the replay buffer.

---
Commit #02b65cc39c0228629336d7d311dcd70ffe3794a7

Now everything is training and evaluating against RuleBasedAgent. Did a bunch of experiments with hyperparameters.

Observation:
- Increasing gamma doesn't seem to do anything (reasonable - we have fixed-length episodes).
- Increasing epsilon doesn't seem to do anything. Policy is still static.
- Reducing batch size doesn't do much (performance slightly worse, training slower)
- Running training every experience (instead of every 8 experiences) doesn't do much (but training is much slower)
- Zeroing out invalid actions doesn't do much (slightly improves training speed).
- Reducing LR from 0.001 to 0.0003 **helps**!! Higher performance, and the policy sometimes becomes non-static in the last 3 tricks.

---
Commit #fcd0575c09e14fdd6718dc2dd13951e5d0fc61a9

Now allowing the agent to take invalid actions and be punished for it. In that case, the agent stays in the same state and receives a negative reward. This results in an endless loop from which the agent can only escape if they change their mind, or eps-greedy exploration kicks in.

Observation:
- At the start of training, the agent is stuck in the loop until the eps-greedy exploration takes them out of it. For now, exploration is limited to valid actions only.
- The agent picks up valid actions very quickly (after few hundred episodes)
- **No more static policies!** Yay! The agent now reacts to every state, in almost all cases a valid action is at the top of the Q-table.
- However, the overall performance is not yet better than previous static policies.
- Most moves seem pretty good, but every now and then there is a glaring mistake.

Discussion:
- This helps against the static policy, but it's not clear if it's required to increase performance.
- After all, the old agent was able to escape the static policy by using a lower LR.

Observation:
- The performance measure (winrate relative to RuleBasedAgent) is skewed and not very representative.
- **Idea**: Since we are now playing against realistic enemies, we can simply use the overall winrate. Also, since both the agent and the enemies are almost completely deterministic, we can reduce the number of samples during evaluation.
- Baseline winrates when playing against 3 other RuleBasedAgents:
    - RuleBasedAgent: 0.421
    - StaticPolicyAgent: 0.316
    - RandomCardAgent: 0.257
        
Next steps:
- Change evaluation criteria to overall winrate.
- Train this with reduced LR and compare.

---
Commit #69bf300b92b285551dc92521a743e40b35e2a687

It's working!! After some parameter tuning, DQNAgent has achieved **super-rule-based** performance with a winrate of **0.513**!

Observation:
- Reducing LR to 0.0003, and further to 0.0001 helps (both with or without invalid actions)
- Increasing gamma to 0.9 helps ALOT when allowing invalid actions (but not when they aren't allowed). 
- Without allowing invalid actions, learning seems to plateau here. Allowing (and punishing) invalid actions is the way forward.

Discussion:
- It's remarkable how well the agent plays with the little information it has. For example, it never sees who actually won a specific trick. Or how many points it has scored. 
- Granted, the RuleBasedAgent also doesn't have that information and plays pretty well. DQNAgent actually has a bit more information (memory of which cards are already gone from the game). A good agent is expected to beat this baseline.
- DQNAgent does some very strong moves:
    1. DQNAgent always opens with a low trump. This seems counter-intuitive, but on second look it's actually genius. By doing this, it is deliberately exploiting a bug in the RuleBasedAgents, causing them to throw away their high trumps (see "TODO: don't schmier an ober!").
    2. Afterwards, it plays trumps in a good order.
    3. If there is a "spatz" (a low non-trump card, which is a liability), it will try to throw it away early in the game. This is a move often done by pros, as keeping the card until the end will usually incur great damage. 
- However, DQNAgent makes 2 clear mistakes:
    1. It loves to play Herz-zehn, even if it's clear that the enemy will beat it. This is always an expensive mistake, as this gives away more points than usual. To me, it's not clear if the consequent loss is properly attributed to this decision. After all, it is causing the enemy to throw away their Trumps, which is desirable. This is a conflict between minimizing damage to self vs maximizing damage to the enemy. I think this may be addressed by including player scores into the state, so the damage done to self can be more easily observed.
    2. Rarely, it tries to keep high trumps (ober) to the last, like a newbie player who is afraid to play them and waits until the enemies have already scored too many points. This move is not completely stupid - DQNAgent may be aiming for a penultimate state where it maxes out on good cards - it knows that this state is often followed by victory. But it actually depends! This too should be easier with scores contained in state, allowing the agent to differentiate between "good cards, good score" and "good cards, bad score".
- Theoretically, the agent could learn to avoid these mistakes in due time, but this might need some more sophisticated exploration strategies.
- **Idea**: Encoding current score into the current state should allow the agent to make that connection much easier. We are *not* encoding score into reward just yet, I don't want to create unnecessary biases for now. 

Overall - the agent is now playing a solo game extremely well!

For now, I think DQNAgent has reached a good state, considering we didn't use any advanced techniques. Only vanilla Q-learning!
After some code cleanup and documentation, I will put the project on hold as I will be quite busy in the next couple of months.

Some "last steps" for DQNAgent:
- Include score into state
- Improve RuleBasedAgent to stop DQNAgent from shamelessly exploiting it
- Train a DQNAgent as non-declaring player, so the user can play against them :)

In the future, I'd like to:
- Try other RL approaches (double DQN, Policy Gradients, latest papers)
- Add memory to RL agents - maybe LSTMs can remember what the other players have done, so we can play with 0 feature engineering?
- Implement MCTS - it should be possible to "solve" Schafkopf with old-fashioned search, as the actions are very limited
- Read up on POMDP, maybe modeling partial observability makes this easier?
- Allow the agents to play all game modes, and include the bidding phase. This should be very interesting.
- Finally - just like in real life, base rewards on money.
- ???
- PROFIT
