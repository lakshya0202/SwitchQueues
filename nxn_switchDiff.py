#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import math
from itertools import permutations
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


# Generate MaxWeight service matrix
def generateMaxWeightServiceMatrix(n, q):
  bipartite_graph = nx.complete_bipartite_graph(n,n)
  s = np.zeros((n,n))
  for e,(u,v) in enumerate(bipartite_graph.edges()):
    bipartite_graph[u][v]['weight'] = q[u][v-n]

  max_weight_match = nx.max_weight_matching(bipartite_graph,maxcardinality=True)
    
  for e,(u,v) in enumerate(max_weight_match):
    s[u][v-n] = 1
  return s

# Given no of inputs and outputs, design a permutation matrix
def generateRandomServiceMatrix(n):
  num_arr = np.arange(n)
  perm_mat = np.zeros((n, n))
  for i in range(n):
    t = np.random.choice(num_arr, 1)[0]
    num_arr = np.delete(num_arr, np.where(num_arr == t))
    perm_mat[i][t] = 1
  return perm_mat

# Given q and d, return max weight service matrix after d random samplings
def generateRandom2ServiceMatrix(n, d, q):
  sum = -1
  service_mat = np.zeros((n, n))
  for i in range(d):
    num_arr = np.arange(n)
    perm_mat = np.zeros((n, n))
    for j in range(n):
      t = np.random.choice(num_arr, 1)[0]
      num_arr = np.delete(num_arr, np.where(num_arr == t))
      perm_mat[i][t] = 1
    t = 0
    # Find the weight
    for j in range(n):
      t += np.inner(perm_mat[i], q[i])
    if t > sum:
      sum = t
      service_mat = perm_mat
  return service_mat

# Choose the max weight service based on previous and present configuration for given q
def generateRandom3ServiceMatrix(n, s_t, q):
  num_arr = np.arange(n)
  perm_mat = np.zeros((n, n))
  for i in range(n):
    t = np.random.choice(num_arr, 1)[0]
    num_arr = np.delete(num_arr, np.where(num_arr == t))
    perm_mat[i][t] = 1
  w1 = 0
  w2 = 0
  for j in range(n):
    w1 += np.inner(s_t[i], q[i])
    w2 += np.inner(perm_mat[i], q[i])
  if w1>w2:
    return s_t
  else:
    return perm_mat

# Return matrix where each individual element is greater than or equal to 0
def positiveMatrix(X):
  for i in range(X.shape[0]):
    for j in range(X.shape[0]):
      if X[i][j] < 0:
        X[i][j] = 0
  return X

# Find the min, so that queue does not hold more than threshold
def elementWiseMin(X, c):
  for i in range(X.shape[0]):
    for j in range(X.shape[0]):
      if X[i][j] > c:
        X[i][j] = c
  return X


# In[3]:


# threshold, max length of each queue
c = 5
# Initialize switch size
n = 3
# seed
random.seed(10)

# Train and test episodes
test_eps = 10000001
train_eps = 10000001

# Arrival rate matrix
a_rate = np.ones((n, n))
a_rate = a_rate*(1/n)*0.99

#a_rate = np.array([[0.2, 0.45], [0.3, 0.25]])
epsilon = 0.1


# Number of possible states for a queued switch with threshold c and number of queues n with n servers: (c+1)\^(n\^2)
# 
# Number of possible actions: n!
# 
# Q: (c+1)^(n^2) * n! size


def chooseEpsilonGreedyServiceConfiguration(n, Q, q, epsilon):
  if np.random.uniform() < epsilon: # when we need to use the greedy policy with respect to Q
    ind = qToList(q)
    Q_value_vector = Q[tuple(ind)]
    res = np.argmax(Q_value_vector)
    return indexToService(n, res)
  else: # when we need to use the random policy
    return generateRandomServiceMatrix(n)

def chooseRandomMaxWeightServiceConfiguration(n, q, epsilon):
  if np.random.uniform() < epsilon: # when we need to use the greedy policy with respect to Q
    return generateMaxWeightServiceMatrix(n, q)
  else: # when we need to use the random policy
    return generateRandomServiceMatrix(n)

# Given queue configuration to list
def qToList(q):
  q = q.astype(int)
  q = q.flatten()
  q = q.tolist();
  return q

def possibleServiceConfigs(n):
  a = ""
  for i in range(n):
    a += str(i)
  p = permutations(a) 

  service_array = []
  for i in p:
    t = ''.join(i)
    service_array.append(t)
  return service_array

# Find the index correlating to the action
def serviceToIndex(n, s):
  service_array = possibleServiceConfigs(n)
  service_index=""
  for i in range(n):
    t = np.where(s[i]==1)[0][0]
    service_index += str(t)
  return service_array.index(service_index)

def indexToService(n, ind):
  service_array = possibleServiceConfigs(n)
  t = service_array[ind]
  s = np.zeros((n, n))
  r = 0
  for i in t:
    s[r, int(i)] = 1
    r += 1
  return s


print("Simulation Results of Differential QLearning")
print()

def DifferentialQLearning(eta, alpha, TRAIN_EPISODES, TEST_EPISODES):
    action_space = math.factorial(n)
    state_space = [c+1]*n*n
    state_space.append(action_space)
    q_switch_matrix = np.zeros((n, n))
    mu = 0
    Q = np.zeros(state_space, dtype=np.float32)

    # Holds the value of the previous service config
    for t in range(1, TRAIN_EPISODES):
        service_config = chooseEpsilonGreedyServiceConfiguration(n, Q, q_switch_matrix, epsilon) # choose service configuration s(t - 1) for the interval [t - 1, t]
        arrival_matrix = np.random.poisson(a_rate) # generate arrival configuration a(t - 1) for the interval [t - 1, t]
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        m_ = mu # store m(t - 1)
        mu = (1 - 1 / t) * mu  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        q_ = q_switch_matrix # store q(t - 1)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))
        ind = serviceToIndex(n, service_config)
        q_index = qToList(q_)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        q_index.append(ind)
        q_index = tuple(q_index)
        # compute past average reward m(t) based on t, m(t - 1), current reward r(t - 1), and Q(q(t), .)
        mu = mu  + eta * alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

        # update action-value estimate Q(q(t - 1), s(t - 1)) based on Q(q(t - 1), s(t - 1)), Q(q(t), .), t, r(t - 1), and m(t - 1)
        Q[q_index] += alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

    avg_reward = 0
    for t in range(1, TEST_EPISODES+1):
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        avg_reward = (1 - 1 / t) * avg_reward  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        # Choose service_config based on current q_switch_matrix through Q Table
        q_index = qToList(q_switch_matrix)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        # Choose vector which gives the highest reward
        ind = np.argmax(Q_value_vector)
        service_config = indexToService(n, ind)
        arrival_matrix = np.random.poisson(a_rate)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))

    print("Average queue length Epsilon Greedy:", avg_reward, "for eta:",  eta, "and alpha:", alpha)
    return avg_reward
                  


# In[6]:


eta = 0.1

alpha = np.array([0.0005, 0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

avg_total_q = np.zeros(len(alpha))
for i in range(len(alpha)):
    avg_total_q[i] = DifferentialQLearning(eta, alpha[i], train_eps, test_eps)

plt.plot(alpha, avg_total_q)
plt.xlabel('Alpha')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = epsilon-greedy, eta = 0.1, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQEpsAlpha.png', bbox_inches='tight')


# In[7]:


# eta = 0.1
eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

alpha = 0.01
# alpha = np.array([0.005, 0.007, 0.008, 0.009, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

avg_total_q = np.zeros(len(eta))
for i in range(len(eta)):
    avg_total_q[i] = DifferentialQLearning(eta[i], alpha, train_eps, test_eps)

plt.plot(eta, avg_total_q)
plt.xlabel('Eta')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = epsilon-greedy, alpha = 0.01, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQEpsEta.png', bbox_inches='tight')


# ### b. Random Sevice Policy

# In[5]:


def DifferentialQLearning(eta, alpha, TRAIN_EPISODES, TEST_EPISODES):
    action_space = math.factorial(n)
    state_space = [c+1]*n*n
    state_space.append(action_space)
    q_switch_matrix = np.zeros((n, n))
    mu = 0
    Q = np.zeros(state_space, dtype=np.float32)

    # Holds the value of the previous service config
    for t in range(1, TRAIN_EPISODES):
        service_config = generateRandomServiceMatrix(n) # choose service configuration s(t - 1) for the interval [t - 1, t]
        arrival_matrix = np.random.poisson(a_rate) # generate arrival configuration a(t - 1) for the interval [t - 1, t]
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        m_ = mu # store m(t - 1)
        mu = (1 - 1 / t) * mu  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        q_ = q_switch_matrix # store q(t - 1)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))
        ind = serviceToIndex(n, service_config)
        q_index = qToList(q_)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        q_index.append(ind)
        q_index = tuple(q_index)
        # compute past average reward m(t) based on t, m(t - 1), current reward r(t - 1), and Q(q(t), .)
        mu = mu  + eta * alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

        # update action-value estimate Q(q(t - 1), s(t - 1)) based on Q(q(t - 1), s(t - 1)), Q(q(t), .), t, r(t - 1), and m(t - 1)
        Q[q_index] += alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

    avg_reward = 0
    for t in range(1, TEST_EPISODES+1):
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        avg_reward = (1 - 1 / t) * avg_reward  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        # Choose service_config based on current q_switch_matrix through Q Table
        q_index = qToList(q_switch_matrix)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        # Choose vector which gives the highest reward
        ind = np.argmax(Q_value_vector)
        service_config = indexToService(n, ind)
        arrival_matrix = np.random.poisson(a_rate)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))

    print("Average queue length Random Policy:", avg_reward, "for eta:",  eta, "and alpha:", alpha)
    return avg_reward
                  


# In[8]:


eta = 0.1

alpha = np.array([0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

avg_total_q = np.zeros(len(alpha))
for i in range(len(alpha)):
    avg_total_q[i] = DifferentialQLearning(eta, alpha[i], train_eps, test_eps)

plt.plot(alpha, avg_total_q)
plt.xlabel('Alpha')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = Random, eta = 0.1, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQRandAlpha.png', bbox_inches='tight')


# In[6]:


# eta = 0.1
eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

alpha = 0.005

avg_total_q = np.zeros(len(eta))
for i in range(len(eta)):
    avg_total_q[i] = DifferentialQLearning(eta[i], alpha, train_eps, test_eps)

plt.plot(eta, avg_total_q)
plt.xlabel('Eta')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = Random, alpha = 0.005, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQRandEta.png', bbox_inches='tight')


# ### c. MaxWeight Policy

# In[5]:


def DifferentialQLearning(eta, alpha, TRAIN_EPISODES, TEST_EPISODES):
    action_space = math.factorial(n)
    state_space = [c+1]*n*n
    state_space.append(action_space)
    q_switch_matrix = np.zeros((n, n))
    mu = 0
    Q = np.zeros(state_space, dtype=np.float32)

    # Holds the value of the previous service config
    for t in range(1, TRAIN_EPISODES):
        service_config = generateMaxWeightServiceMatrix(n, q_switch_matrix) # choose service configuration s(t - 1) for the interval [t - 1, t]
        arrival_matrix = np.random.poisson(a_rate) # generate arrival configuration a(t - 1) for the interval [t - 1, t]
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        m_ = mu # store m(t - 1)
        mu = (1 - 1 / t) * mu  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        q_ = q_switch_matrix # store q(t - 1)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))
        ind = serviceToIndex(n, service_config)
        q_index = qToList(q_)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        q_index.append(ind)
        q_index = tuple(q_index)
        # compute past average reward m(t) based on t, m(t - 1), current reward r(t - 1), and Q(q(t), .)
        mu = mu  + eta * alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

        # update action-value estimate Q(q(t - 1), s(t - 1)) based on Q(q(t - 1), s(t - 1)), Q(q(t), .), t, r(t - 1), and m(t - 1)
        Q[q_index] += alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

    avg_reward = 0
    for t in range(1, TEST_EPISODES+1):
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        avg_reward = (1 - 1 / t) * avg_reward  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        # Choose service_config based on current q_switch_matrix through Q Table
        q_index = qToList(q_switch_matrix)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        # Choose vector which gives the highest reward
        ind = np.argmax(Q_value_vector)
        service_config = indexToService(n, ind)
        arrival_matrix = np.random.poisson(a_rate)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))

    print("Average queue length MaxWeight Policy:", avg_reward, "for eta:",  eta, "and alpha:", alpha)
    return avg_reward
                  


# In[8]:


eta = 0.1
# eta = np.linspace(0, 2, 21)

alpha = np.array([0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

avg_total_q = np.zeros(len(alpha))
for i in range(len(alpha)):
    avg_total_q[i] = DifferentialQLearning(eta, alpha[i], train_eps, test_eps)

plt.plot(alpha, avg_total_q)
plt.xlabel('Alpha')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = Max Weight, eta = 0.1, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQMWAlpha.png', bbox_inches='tight')


# In[7]:


# eta = 0.1
eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

alpha = 0.7

avg_total_q = np.zeros(len(eta))
for i in range(len(eta)):
    avg_total_q[i] = DifferentialQLearning(eta[i], alpha, train_eps, test_eps)

plt.plot(eta, avg_total_q)
plt.xlabel('Eta')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = Max Weight, alpha = 0.7, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQMWEta.png', bbox_inches='tight')


# ### d. ε MaxWeight Random Policy

# In[8]:


def DifferentialQLearning(eta, alpha, TRAIN_EPISODES, TEST_EPISODES, epsilon):
    action_space = math.factorial(n)
    state_space = [c+1]*n*n
    state_space.append(action_space)
    q_switch_matrix = np.zeros((n, n))
    mu = 0
    Q = np.zeros(state_space, dtype=np.float32)

    # Holds the value of the previous service config
    for t in range(1, TRAIN_EPISODES):
        service_config = chooseRandomMaxWeightServiceConfiguration(n, q_switch_matrix, epsilon) # choose service configuration s(t - 1) for the interval [t - 1, t]
        arrival_matrix = np.random.poisson(a_rate) # generate arrival configuration a(t - 1) for the interval [t - 1, t]
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        m_ = mu # store m(t - 1)
        mu = (1 - 1 / t) * mu  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        q_ = q_switch_matrix # store q(t - 1)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))
        ind = serviceToIndex(n, service_config)
        q_index = qToList(q_)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        q_index.append(ind)
        q_index = tuple(q_index)
        # compute past average reward m(t) based on t, m(t - 1), current reward r(t - 1), and Q(q(t), .)
        mu = mu  + eta * alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

        # update action-value estimate Q(q(t - 1), s(t - 1)) based on Q(q(t - 1), s(t - 1)), Q(q(t), .), t, r(t - 1), and m(t - 1)
        Q[q_index] += alpha * (r - m_ + min(Q_value_vector) - Q[q_index])

    avg_reward = 0
    for t in range(1, TEST_EPISODES+1):
        r = np.sum(q_switch_matrix) # compute current reward r(t - 1)
        avg_reward = (1 - 1 / t) * avg_reward  + 1 / t * r # compute past average reward m(t) based on t, m(t - 1), and current reward r(t - 1)
        # Choose service_config based on current q_switch_matrix through Q Table
        q_index = qToList(q_switch_matrix)
        tup_q_i = tuple(q_index)
        Q_value_vector = Q[tup_q_i]
        # Choose vector which gives the highest reward
        ind = np.argmax(Q_value_vector)
        service_config = indexToService(n, ind)
        arrival_matrix = np.random.poisson(a_rate)
        q_switch_matrix = positiveMatrix(elementWiseMin(q_switch_matrix + arrival_matrix - service_config, c))

    print("Average queue length Epsilon MaxWeight Random:", avg_reward, "for eta:",  eta, ",alpha:", alpha, "and epsilon:", epsilon)
    return avg_reward
                  


# Taking the best value of $\eta$ and $\alpha$ from above

# In[9]:


# Taking best eta and alpha values from the above simulations
eta = 0.9
alpha = 0.01
Epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
avg_total_q = np.zeros(len(Epsilon))
for i in range(len(Epsilon)):
  avg_total_q[i] = DifferentialQLearning(eta, alpha, train_eps, test_eps, Epsilon[i])

plt.plot(Epsilon, avg_total_q)
plt.xlabel('Epsilon')
plt.ylabel('Average Total Queue Length')
plt.title('Differential Q-learning with Behavior Policy = epsilon MaxWeight+Random, eta = 0.6, alpha = 0.6, and numOfIterations = %d' %train_eps)
plt.grid()
plt.savefig('DiffQRMWEps.png', bbox_inches='tight')