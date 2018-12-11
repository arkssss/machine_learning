from Markov_with_action import MarkovWithAction

statements = ['S1', 'S2', 'S3']
rewords = {
    'S1': -1,
    'S2': 2,
    'S3': 6
}
transformer = {
    'S1': {
        'A': {
            'S1': 0.1,
            'S2': 0.9,
            'S3': 0
        },
        'B': {
            'S1': 0.2,
            'S2': 0.6,
            'S3': 0.2
        },
        'C': {
            'S1': 0.7,
            'S2': 0,
            'S3': 0.3
        }
    },
    'S2': {
        'A': {
            'S1': 0,
            'S2': 1,
            'S3': 0
        },
        'B': {
            'S1': 1,
            'S2': 0,
            'S3': 0
        },
        'C': {
            'S1': 0,
            'S2': 0.6,
            'S3': 0.4
        }
    },
    'S3': {
        'A': {
            'S1': 0.7,
            'S2': 0,
            'S3': 0.3
        },
        'B': {
            'S1': 0.9,
            'S2': 0,
            'S3': 0.1
        },
        'C': {
            'S1': 0,
            'S2': 1,
            'S3': 0
        }
    },

}
actions = ['A', 'B', 'C']
# 20 + 0.9*(12+12+2) = 20 + 26*0.9 =
discount = 0.9
# 初始化
Markov_one = MarkovWithAction(statements, rewords, transformer, discount, actions)
# print(Markov_one.compute_j(30))
# 计算
res = Markov_one.compute_j(30)
# for item in Markov_one.compute_j(30):
#     print(item.values())
# 存入文件
file_name = "data_with_action.txt"
# 4为保留后四位
Markov_one.store_as_file(file_name, res, 4)


