from utils import str_key, display_dict
from utils import set_prob, set_reward, get_prob, get_reward
from utils import set_value, set_pi, get_value, get_pi

S = ['浏览手机中', '第一节课', '第二节课', '第三节课', '休息中']
A = ['浏览手机', '学习', '离开浏览', '泡吧', '退出学习']
R = {}
P = {}
gamma = 1.0

set_prob(P, S[0], A[0], S[0])  # 浏览手机中 - 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1])  # 浏览手机中 - 离开浏览 -> 第一节课
set_prob(P, S[1], A[0], S[0])  # 第一节课 - 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2])  # 第一节课 - 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3])  # 第二节课 - 学习 -> 第三节课
set_prob(P, S[2], A[4], S[4])  # 第二节课 - 退出学习 -> 退出休息
set_prob(P, S[3], A[1], S[4])  # 第三节课 - 学习 -> 退出休息
set_prob(P, S[3], A[3], S[1], p = 0.2)  # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[2], p = 0.4)  # 第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[3], p = 0.4)  # 第三节课 - 泡吧 -> 第一节课

set_reward(R, S[0], A[0], -1)  # 浏览手机中 - 浏览手机 -> -1
set_reward(R, S[0], A[2],  0)  # 浏览手机中 - 离开浏览 -> 0
set_reward(R, S[1], A[0], -1)  # 第一节课 - 浏览手机 -> -1
set_reward(R, S[1], A[1], -2)  # 第一节课 - 学习 -> -2
set_reward(R, S[2], A[1], -2)  # 第二节课 - 学习 -> -2
set_reward(R, S[2], A[4],  0)  # 第二节课 - 退出学习 -> 0
set_reward(R, S[3], A[1], 10)  # 第三节课 - 学习 -> 10
set_reward(R, S[3], A[3], +1)  # 第三节课 - 泡吧 -> -1

MDP = (S, A, R, P, gamma)

Pi = {}
set_pi(Pi, S[0], A[0], 0.5) # 浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5) # 浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5) # 第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5) # 第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5) # 第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5) # 第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5) # 第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5) # 第三节课 - 泡吧


def compute_q(MDP, V, s, a):
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa


def compute_v(MDP, V, Pi, s):
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)
    return v_s


def update_V(MDP, V, Pi):
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime


def policy_evaluate(MDP, V, Pi, n):
    for i in range(n):
        V = update_V(MDP, V, Pi)
        #display_dict(V)
    return V


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
        V_prime[str_key(s)] = compute_v_from_max_q(MDP, V_prime, s)
    return V_prime


def value_iterate(MDP, V, n):
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


if __name__ == '__main__':
    print("----状态转移概率字典（矩阵）信息:----")
    display_dict(P)
    print("----奖励字典（函数）信息:----")
    display_dict(R)
    print("----状态转移概率字典（矩阵）信息:----")
    display_dict(Pi)
    # 初始时价值为空，访问时会返回0
    V = {}
    # V = policy_evaluate(MDP, V, Pi, 100)
    V_star = value_iterate(MDP, V, 4)
    display_dict(V_star)
