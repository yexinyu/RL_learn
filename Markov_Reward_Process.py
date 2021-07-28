import numpy as np

num_states = 7
# {"0": "C1", "1": "C2", "2": "C3", "3": "Pass", "4": "Pub", "5": "FB", "6": "Sleep" }
i_to_n = {}
i_to_n["0"] = "C1"
i_to_n["1"] = "C2"
i_to_n["2"] = "C3"
i_to_n["3"] = "Pass"
i_to_n["4"] = "Pub"
i_to_n["5"] = "FB"
i_to_n["6"] = "Sleep"

n_to_i = {}
for i, name in zip(i_to_n.keys(), i_to_n.values()):
    n_to_i[name] = int(i)

Pss = [
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
    [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
]

Pss = np.array(Pss)
rewards = [-2, -2, -2, 10, 1, -1, 0]
gama = 0.5


def compute_return(start_index=0, chain=None, gamma=0.5) -> float:
    retrn, power, gamma = 0.0, 0, gamma
    for i in range(start_index, len(chain)):
        retrn += np.power(gamma, power) * rewards[n_to_i[chain[i]]]
        power += 1
    return retrn


def compute_value(Pss, rewards, gamma=0.5):
    rewards = np.array(rewards).reshape(-1, 1)
    values = np.dot(np.linalg.inv(np.eye(7,7) - gamma*Pss), rewards)
    return values


if __name__ == '__main__':
    values = compute_value(Pss, rewards, gamma=0.99999)
    print(values)
