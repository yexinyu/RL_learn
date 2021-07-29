# -*- coding: UTF-8 -*-

S = [i for i in range(16)]  # State Space
A = ['n', 'e', 's', 'w']  # Action Space
ds_actions = {'n': -4, 'e': 1, 's': 4, 'w': -1}

# P,R will be generated from dynamics
def dynamics(s, a):
    s_prime = s
    if (s%4 == 0 and a == 'w') or (s < 4 and a == 'n') \
            or ((s+1)%4 == 0 and a == 'e') or (s > 11 and a == 's') \
            or s in [0,15]:
        pass
    else:
        ds = ds_actions[a]
        s_prime = s + ds
    reward = 0 if s in [0, 15] else -1
    is_end = True if s in [0, 15] else False
    return s_prime, reward, is_end


def P(s, a, s1):
    s_prime, _, _ = dynamics(s, a)
    return s1 == s_prime


def R(s, a):
    _, r, _ = dynamics(s, a)
    return r


gamma = 1.00
MDP = (S, A, R, P, gamma)

def uniform_random_pi(MDP = None, V = None, s = None, a = None):
    _, A, _, _, _ = MDP
    n = len(A)
    return 0 if n == 0 else 1.0/n


def greedy_pi(MDP, V, s, a):
    S, A, P, R, gamma = MDP
    max_v, a_max_v = -float('inf'), []
    for a_opt in A:
        s_prime, reward, _ = dynamics(s, a_opt)
        v_s_prime = get_value(V, s_prime)
        if v_s_prime > max_v:
            max_v = v_s_prime
            a_max_v = [a_opt]
        elif v_s_prime == max_v:
            a_max_v.append(a_opt)
    n = len(a_max_v)
    if n == 0:
        return 0.0
    return 1.0/n if a in a_max_v else 0.0

def get_pi(Pi, s, a, MDP = None, V = None):
    return Pi(MDP, V, s, a)


def get_prob(P, s, a, s1):
    return P(s, a, s1)


def get_reward(R, s, a):
    return R(s, a)


def set_value(V, s, v):
    V[s] = v


def get_value(V, s):
    return V[s]


def display_V(V):
    for i in range(16):
        print('{0:>6.2f}'.format(V[i]),end = " ")
        if (i+1) % 4 == 0:
            print("")


def compute_q(MDP, V, s, a):
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s, a) + gamma * q_sa
    return  q_sa


def compute_v(MDP, V, Pi, s):
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return v_s


def update_V(MDP, V, Pi):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v(MDP, V_prime, Pi, s))
    return V_prime


def policy_evaluate(MDP, V, Pi, n):
    for i in range(n):
        V = update_V(MDP,V, Pi)
    return V


def policy_iterate(MDP, V, Pi, n, m):
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi = greedy_pi
    return  V


def compute_v_from_max_q(MDP, V, s):
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa >= v_s:
            v_s = qsa
    return v_s


def update_V_without_pi(MDP, V):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
    return V_prime


def value_iterate(MDP, V, n):
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


if __name__ == '__main__':
    V = [0 for _ in range(16)]
    V_pi = policy_evaluate(MDP, V, uniform_random_pi, 100)
    display_V(V_pi)
    V = [0 for _ in range(16)]
    V_pi = policy_evaluate(MDP, V, greedy_pi, 100)
    display_V(V_pi)
    V = [0 for _ in range(16)]
    V_pi = policy_iterate(MDP, V, greedy_pi, 1, 100)
    display_V(V_pi)
    V_star = value_iterate(MDP, V, 4)
    display_V(V_star)