# -*- coding: UTF-8 -*-

from random import shuffle
from queue import Queue
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from chap04.utils04 import str_key, set_dict, get_dict


class Gamer():
    def __init__(self, name = "", A = None, display = False):
        self.name = name
        self.cards = []
        self.display = display
        self.policy = None
        self.learning_method = None
        self.A = A


    def __str__(self):
        return self.name


    def _value_of(self, card):
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J', 'Q', 'K']:
                v = 10
            else:
                v = 0
        finally:
            return v


    def get_points(self):
        num_of_useable_ace = 0
        total_point = 0
        cards = self.cards
        if cards is None:
            return 0, False
        for card in cards:
            v = self._value_of(card)
            if v == 1:
                num_of_useable_ace += 1
                v = 11
            total_point += v
        while total_point > 21 and num_of_useable_ace > 0:
            total_point -= 10
            num_of_useable_ace -= 1
        return total_point, bool(num_of_useable_ace)


    def receive(self, cards = []):
        cards = list(cards)
        for card in cards:
            self.cards.append(card)


    def discharge_cards(self):
        self.cards.clear()


    def cards_info(self):
        self._info("{}{}现 在 的 牌:{}\n".format(self.role, self,self.cards))


    def _info(self, msg):
        if self.display:
            print(msg, end="")


class Dealer(Gamer):
    def __init__(self, name = "", A = None, display = False):
        super(Dealer, self).__init__(name, A, display)
        self.role = "Dealer"
        self.policy = self.dealer_policy


    def first_card_value(self):
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])


    def dealer_policy(self, Dealer = None):
        action = ""
        dealer_points, _ = self.get_points()
        if dealer_points >= 17:
            action = self.A[1]
        else:
            action = self.A[0]
        return action


class Player(Gamer):
    def __init__(self, name = "", A = None, display = False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy
        self.role = "Player"


    def get_state(self, dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return dealer_first_card_value, player_points, useable_ace


    def get_state_name(self, dealer):
        return str_key(self.get_state(dealer))


    def naive_policy(self, dealer = None):
        player_points, _ = self.get_points()
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action


class Arena():
    def __init__(self, display = None, A = None):
        self.cards = ['A','2','3','4','5','6','7','8','9','10','J','Q',"K"]*4
        self.card_q = Queue(maxsize = 52)
        self.cards_in_pool = []
        self.display = display
        self.episodes = []
        self.load_cards(self.cards)
        self.A = A


    def load_cards(self, cards):
        shuffle(cards)
        for card in cards:
            self.card_q.put(card)
        cards.clear()
        return


    def reward_of(self, dealer, player):
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, player_points, dealer_points, useable_ace


    def serve_card_to(self, player, n = 1):
        cards = []
        for _ in range(n):
            if self.card_q.empty():
                self._info("\n发 牌 器 没 牌 了 , 整 理 废 牌 , 重 新 洗 牌;")
                shuffle(self.cards_in_pool)
                self._info("一 共 整 理 了{}张 已 用 牌 , 重 新 放 入 发 牌 器\n".format(\
                    len(self.cards_in_pool)))
                assert(len(self.cards_in_pool) > 20)
                self.load_cards(self.cards_in_pool)
            cards.append(self.card_q.get())
        self._info("发 了{}张 牌({})给{}{};".format(n, cards, player.role, player))
        player.receive(cards)
        player.cards_info()

    def _info(self, message):
        if self.display:
            print(message, end="")

    def recycle_cards(self, *players):
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards()

    def play_game(self, dealer, player):
        self._info("===== start a new round =====\n")
        self.serve_card_to(player, n = 2)
        self.serve_card_to(dealer, n = 2)
        episode = []
        if player.policy is None:
            self._info("Player needs a policy")
            return
        if dealer.policy is None:
            self._info("Dealer needs a policy")
            return
        while True:
            action = player.policy(dealer)
            self._info("{}{}选 择:{};".format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action))
            if action == self.A[0]:
                self.serve_card_to(player)
            else:
                break

        reward, player_points, dealer_points, useable_ace = self.reward_of(\
            dealer, player)

        if player_points > 21:
            self._info("玩 家 爆 点{}输 了 , 得 分:{}\n".format(player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward))
            self._info("===== end of this round =====")
            return episode, reward

        self._info("\n")
        while True:
            action = dealer.policy()
            self._info("{}{}选 择:{};".format(dealer.role, dealer, action))
            if action == self.A[0]:
                self.serve_card_to(dealer)
            else:
                break
        self._info("\n双 方 均 停 止 叫 牌;\n")

        reward, player_points, dealer_points, useable_ace = self.reward_of( \
            dealer, player)
        player.cards_info()
        dealer.cards_info()
        if reward == +1:
            self._info("Player Win!")
        elif reward == -1:
            self._info("Player Lose!")
        else:
            self._info("Tied!")
        self._info("玩 家{}点 ,庄 家{}点\n".format(player_points, dealer_points))
        self._info("===== end of this round =====")
        self.recycle_cards(player, dealer)
        self.episodes.append((episode, reward))

        return episode, reward


    def play_games(self, dealer, player, num = 2, show_statistic = True):
        results = [0, 0, 0]
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)

        if show_statistic:
            print("共 玩 了{} 局 , 玩 家 赢{} 局 , 和{} 局 , 输{} 局 , 胜 率 : {:.2f},不 输 率:{:.2f}"\
                  .format(num, results[2],results[1],results[0],results[2]/num,(results[2]+results[1])/num))
            return


    def _info(self, message):
        if self.display:
            print(message, end="")


def policy_evaluate(episodes, V , Ns):
    for episode, r in episodes:
        for s, a in episode:
            ns = get_dict(Ns, s)
            v = get_dict(V, s)
            set_dict(Ns, ns + 1, s)
            set_dict(V, v + (r - v)/(ns + 1), s)


def draw_value(value_dict, useable_ace = True, is_q_dict = False, A = None):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(1, 11, 1)
    y = np.arange(12, 22, 1)
    X, Y = np.meshgrid(x, y)
    row, col = X.shape
    Z = np.zeros((row, col))
    if is_q_dict:
        n = len(A)
    for i in range(row):
        for j in range(col):
            state_name = str(X[i,j])+"_"+str(Y[i,j])+"_"+str(useable_ace)
            if not is_q_dict:
                Z[i, j] = get_dict(value_dict, state_name)
            else:
                assert(A is not None)
                for a in A:
                    new_state_name = state_name + "_" + str(a)
                    q = get_dict(value_dict, new_state_name)
                    if q >= Z[i, j]:
                        Z[i, j] = q
    ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, color = "lightgray")
    plt.show()

V = {}
Ns = {}


if __name__ == '__main__':
    A = ["twist", "stick"]
    display = False
    player = Player(A = A, display = display)
    dealer = Dealer(A = A, display = display)
    arena = Arena(A = A, display = display)
    arena.play_games(dealer, player, num = 200000)
    policy_evaluate(arena.episodes, V, Ns)
    draw_value(V, useable_ace=True, A=A)
    draw_value(V, useable_ace=False, A=A)