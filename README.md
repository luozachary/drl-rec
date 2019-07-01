# drl-rec
Implement deep reinforcement learning(DDPG) for recommendation system.

### data format
train data

|state          |action    |reward      |n_state            |recall     |
| --------------| :-------:| :---------:| :----------------:|:---------:|
|id1,id2,id3,id4|id5       |1.0         |id1,id2,id3,id4,id5|id5,id6    |

recall data

|item_id    |embedding          |
| --------- | :----------------:|
|id1        |`[0.123,0.345,0.421]`|


### reference
[Deep Reinforcement Learning for List-wise Recommendations](https://arxiv.org/abs/1801.00209)

