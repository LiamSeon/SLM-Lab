from collections import namedtuple, deque
import random

# Deep Q Laerning => L1 Loss(Policy Output, Target Output * gamma + reward)
# Buffer : state, action, next state, reward
# reward : 특정 actions에 대한 reward를 return
# 즉, Deep Q Learning의 PolcyNet과 TargetNet은 일종의 Approximator일 뿐이다.
# Inferece Phase : State -> Action (Policy Networks)
# Training Phase : State -> Action, 여기서 가장 큰 Reward를 구한다. 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)