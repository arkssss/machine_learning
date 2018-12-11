"""
a Markov System with action
FangZhou 2018124048
"""
STATE_TYPE = ['S1', 'S2', 'S3', 'S4', 'S5']     # 状态的类型
REWORD = {
    "S1":  1.0,
    "S2": -1.0,
    "S3":  0.0,
    "S4":  3.0,
    "S5":  1.0
}
ACTION = ["A", "B", "C", "D"]   # 事件名
PROBABILITY = {                 # 转移概率
    "S1": {
        "A": {
            "S1": 0.2,
            "S2": 0.8,
            "S3": 0,
            "S4": 0,
            "S5": 0
        },
        "B": {
            "S1": 0,
            "S2": 0,
            "S3": 1,
            "S4": 0,
            "S5": 0,
        },
        "C": {
            "S1": 1,
            "S2": 0,
            "S3": 0,
            "S4": 0,
            "S5": 0,
        },
        "D": {
            "S1": 0.8,
            "S2": 0,
            "S3": 0.2,
            "S4": 0,
            "S5": 0,
        }
    },
    "S2": {
        "A": {
            "S1": 0.7,
            "S2": 0,
            "S3": 0,
            "S4": 0.3,
            "S5": 0
        },
        "B": {
            "S1": 0,
            "S2": 0.1,
            "S3": 0,
            "S4": 0.9,
            "S5": 0,
        },
        "C": {
            "S1": 0,
            "S2": 0,
            "S3": 0.8,
            "S4": 0.2,
            "S5": 0,
        },
        "D": {
            "S1": 0.8,
            "S2": 0.2,
            "S3": 0,
            "S4": 0,
            "S5": 0,
        }
    },
    "S3": {
        "A": {
            "S1": 0,
            "S2": 0.5,
            "S3": 0.5,
            "S4": 0,
            "S5": 0
        },
        "B": {
            "S1": 0,
            "S2": 0.5,
            "S3": 0.5,
            "S4": 0,
            "S5": 0,
        },
        "C": {
            "S1": 0,
            "S2": 0,
            "S3": 1,
            "S4": 0,
            "S5": 0,
        },
        "D": {
            "S1": 0.5,
            "S2": 0,
            "S3": 0.5,
            "S4": 0,
            "S5": 0,
        }
    },
    "S4": {
        "A": {
            "S1": 0,
            "S2": 0,
            "S3": 0,
            "S4": 1,
            "S5": 0
        },
        "B": {
            "S1": 0,
            "S2": 0.3,
            "S3": 0,
            "S4": 0.7,
            "S5": 0,
        },
        "C": {
            "S1": 0,
            "S2": 0,
            "S3": 0,
            "S4": 0,
            "S5": 1,
        },
        "D": {
            "S1": 0,
            "S2": 0,
            "S3": 0,
            "S4": 0.5,
            "S5": 0.5,
        }
    },
    "S5": {
        "A": {
            "S1": 0,
            "S2": 0,
            "S3": 0.3,
            "S4": 0,
            "S5": 0.7
        },
        "B": {
            "S1": 0,
            "S2": 0,
            "S3": 0,
            "S4": 0.6,
            "S5": 0.4,
        },
        "C": {
            "S1": 0,
            "S2": 0,
            "S3": 0,
            "S4": 0.9,
            "S5": 0.1,
        },
        "D": {
            "S1": 0.8,
            "S2": 0,
            "S3": 0,
            "S4": 0,
            "S5": 0.2,
        }
    }
}
EPSILON = 0.0001    # 差值
GAMMA = 0.9         # 衰减

def markov():
    """
    :return: 一个list形式的J向量 e.g
    [
        {
            's1' : xxx,
            's2' : xxx,
            ...
        },
        {
            's1' : xxx,
            's2' : xxx,
            ...
        }
        ...
    ]
    """
    J = list()          # answer
    J.append(REWORD)    # 添加初始状态
    gap = 1             # 初始化n和n+1年之间的J的差值,gap 应该为几个状态间的最大差值
    while gap > EPSILON:
        temp_map = dict()
        last_J = J[-1:][0]     # 上一个J状态reword向量, 用于更新此时的J状态
        for each_state in STATE_TYPE:
            # 按state更新J
            max_j_with_action = -10000
            for each_action in ACTION:
                tmp_j_with_action = 0
                for sta, pro in PROBABILITY[each_state][each_action].items():
                    tmp_j_with_action += last_J[sta] * pro
                if tmp_j_with_action > max_j_with_action:
                    max_j_with_action = tmp_j_with_action
            update_num = max_j_with_action * GAMMA + REWORD[each_state]
            if abs(update_num - last_J[each_state]) > gap:
                gap = abs(update_num - last_J[each_state])   # 更新gap
            temp_map[each_state] = update_num
        gap = get_gap(temp_map, last_J)
        J.append(temp_map)
    return J


def get_gap(next_dict, last_dict):
    """
    获取next_dict和last_dict之间的最大gap
    :param next_dict:
    :param last_dict:
    :return:
    """
    gap_reword = list()
    for state, reword in next_dict.items():
        gap_reword.append(abs(last_dict[state] - reword))
    return max(gap_reword)


if __name__ == "__main__":
    ans_J = markov()
    for item in ans_J:
        print(item)

































