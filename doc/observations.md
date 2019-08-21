Writing down random ideas and observations.

Situation:
- Commit #5c34e5723726bb9c62c6a7ffb92ce101e13ba409
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
- Goal: Revert to linear (because this is the "correct" q-learning paradigm), but tune something else to get the same good learning.4