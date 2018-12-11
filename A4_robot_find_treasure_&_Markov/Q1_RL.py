"""
  take Reinforcement Learning in two-dimension Array based on Q-Learning
  Lead the robot training for several times, and the robot can find the shortest path
  FangZhou 2018124048
"""
import random
# some global var
MAX_EPISODES = 200   # 学习的次数
GOLD_STATE = (5, 4)     # 目标状态
INIT_STATE = (7, 3)     # 初始状态
GOLD_REWORD = 10
STATES = [(2, 6), (2, 5), (2, 4),
          (2, 3), (2, 2), (3, 6),
          (3, 3), (3, 2), (4, 6), (4, 5), (4, 4),
          (4, 2), (5, 6), (5, 4), (5, 2), (6, 6), (6, 3),
          (6, 2), (7, 6), (7, 5), (7, 4), (7, 3), (7, 1),
          (8, 2), (8, 1)]        # 状态的形式,只有在这些状态中的位置, 才能被robot访问。 为list形式的
ACTIONS = ["up", "down", "left", "right"]  # 动作分类， 为list形式
GREEDY_RATE = 0.9    # 贪婪值，控制着在选则行为的时候，有0.9的概率会按值选择，有0.1的概率会随机选择
ALPHA = 0.01     # 学习速率
GAMMA = 0.95     # 衰减因子，控制着之后的reword对此刻action选择的影响衰减比例


def init_q_table(the_states, the_actions):
    """
    根据the_states ，the_actions 来初始化一个q_table
    q_table的形式为:
    {
        (2, 6):         #不管状态(2, 6)能否采取这种形式的action，都先初始化
        {
            "up" : xxx
            "down" : xxx
            "left" : xxx
            "right" : xxx
        }
        ...
    }
    :return: 一个Qtable
    """
    my_q_table = dict()
    for state in the_states:
        my_q_table[state] = dict()
        for action in the_actions:
            my_q_table[state][action] = 0.00
    return my_q_table


def choose_action(q_table, current_state, greedy_rate = GREEDY_RATE):
    """
    根据一个q_table 来选择 current_state下的应该选择的action
    :param q_table:
    :param current_state: 当前的状态
    :return: 一个action
    """
    # 在10%的概率下 或者 当前q_table对应的全为0的情况下， 则随机选择一个action
    if random.randint(0, 100) > greedy_rate * 100 or all_zero(q_table[current_state]):
        action_length = len(ACTIONS)
        return ACTIONS[random.randint(0, action_length-1)]
    else:
        # 此时为按值选取
        ac, val = max_action_val(q_table[current_state])
        return ac


def get_env_feedback(current_state, action):
    """
    这个feedback并不是q_table里面的值
    :param state:
    :param action:
    :return: 返回在current_state下采取action后获得的reword， 以及下一步的state
    """
    x_axis = current_state[0]
    y_axis = current_state[1]
    if action == "up":
        # 往上走
        y_axis += 1
    elif action == "down":
        # 往下走
        y_axis -= 1
    elif action == "left":
        # 往左走
        x_axis -= 1
    else:
        # 往右走
        x_axis += 1
    next_state = (x_axis, y_axis)
    if next_state not in STATES:
        # 撞到墙壁，状态不变
        return 0, current_state
    else:
        # 没有撞到墙壁
        if is_terminal(next_state):
            return GOLD_REWORD, next_state
        else:
            return 0, next_state


def is_terminal(the_state):
    """判断此时是不是终点"""
    if the_state == GOLD_STATE:
        return True
    else:
        return False


def max_action_val(state_action):
    """
    选择出这个dict中Reword最大的action
    :param q_table: 
    :param current_state: 
    :return: 最大的值对应的action, 以及相应的val
    """
    max_key = 0
    max_val = 0
    for key, val in state_action.items():
        if not max_key:
            max_key = key
            max_val = val
            continue
        if val > max_val:
            max_key = key
            max_val = val
    return max_key, max_val


def all_zero(state_action):
    """
    :param state_action: 传入当前的state的对应的q_table ,形式如下
        {
            "top" : 0.00,
            "down" : 0.00,
            "left" : 0.00,
            "right" : 0.00
        }
    :return: true(全0) or false(非全0)
    """
    for val in state_action.values():
        if val:
            return False
    return True


def rl():
    """
    主循环函数
    :return:
    """
    q_table = init_q_table(STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        s = INIT_STATE  # 初始化状态
        step = 0
        while not is_terminal(s):
            # 在还不是终点的时候
            action = choose_action(q_table, s)
            R, next_s = get_env_feedback(s, action)
            ac, val = max_action_val(q_table[next_s])
            # 更新Q(s, a)值
            gap = R + GAMMA * val - q_table[s][ac]
            q_table[s][action] += ALPHA * gap
            s = next_s
            step += 1
        print("episode : "+str(episode)+" reach gold steps: " + str(step))
    return q_table


def get_final_path(q_table):
    """
    根据最后的q_table, 得出最后的path
    :param q_table:
    :return:
    """
    s = INIT_STATE
    print(s)
    while not is_terminal(s):
        action = choose_action(q_table, s, 1)
        R, next_s = get_env_feedback(s, action)
        s = next_s
        print(s)


if __name__ == "__main__":
    q_table = rl()
    print("the final path is: ")
    get_final_path(q_table)




















