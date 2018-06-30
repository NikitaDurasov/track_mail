import gym
import pickle
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T 

env = gym.make('Breakout-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Положить переход в память."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Получить сэмпл из памяти """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN алгоритм
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3136, 512)
        self.head = nn.Linear(512, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(F.relu(self.fc(x.view(x.size(0), -1))))

BATCH_SIZE = 16
GAMMA = 0.99
EPS_DECAY = 20000
CAPACITY = 15000

IM_SIZE = 84
EPS_START = 0.95
EPS_END = 0.05

crop_im_start, crop_im_end = 35, 195

resize = T.Compose([T.ToPILImage(),
                    T.Resize(IM_SIZE, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = screen[:, crop_im_start:crop_im_end, :]
    screen = 0.2126*screen[0] + 0.7152*screen[1] + 0.0722*screen[2]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen).unsqueeze(0)
    return resize(screen).unsqueeze(0).type(Tensor)

MOMENTUM = 0.95
LR = 1e-4

model = DQN()

if use_cuda:
    model.cuda()

memory = ReplayMemory(15000)

optimizer = optim.RMSprop(model.parameters(), lr=LR, momentum=MOMENTUM)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1
, 1)
    else:
        return LongTensor([[random.randrange(4)]])

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    # выбираем новый батч
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Для всех состояний считаем маску не финальнсти и конкантенируем их
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # Блокируем прохождение градиента для вычисления функции ценности действия
    # volatile=True
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Считаем Q(s_t, a) - модель дает Q(s_t), затем мы выбираем
    # колоки, которые соответствуют нашим действиям на щаге
    state_action_values = model(state_batch).gather(1, action_batch)

    # Подсчитываем ценность состяония V(s_{t+1}) для всех последующмх состояний.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0] # берем значение максимума

    # Для подсчета лоса нам нужно будет разрешить прохождение градиента по переменной
    # блокировку, которого мы унаследовали
    # requires_grad=False

    next_state_values.volatile = False
    # Считаем ожидаемое значение функции оценки ценности действия  Q-values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Считаем ошибку Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    #print("LOSS: {0}".format(loss.data[0]))
    # Оптимизация модели
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



num_episodes = 4000

no_action = 0

for i_episode in tqdm(range(num_episodes)):
    if i_episode % 10 == 0 and i_episode != 0:
        print("Average reward for the last 10 episodes:", sum_rewards / 10)
    # Инициализация среды

    total_rewards = 0.0
    env.reset()

    last_four_screens = [get_screen() for i in range(4)]

    state = torch.cat(last_four_screens, dim=1)

    for t in count():
        # Выбрать и выполнить новое действие
        action = select_action(state)

        if action[0, 0] == 0:
            no_action += 1

        if no_action > 20:
            action[0][0] = 1 + random.randint(0, 2)
            no_action = 0

        _, reward, done, _ = env.step(action[0, 0])
        total_rewards += reward
        reward = Tensor([reward])

        # Получаем новое состояние
        last_four_screens.pop(0)
        last_four_screens.append(get_screen())

        if not done:
            next_state = torch.cat(last_four_screens, dim=1)
        else:
            next_state = None

        # Сохраняем состояние, следующее состояние, награду и действие в память
        memory.push(state, action, next_state, reward)

        # Переходим в новое состояние
        state = next_state

        # Шаг оптимизации
        optimize_model()
        if done:
            break

env.close()

with open('breakout_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Complete')
