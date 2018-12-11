import os
class MarkovWithAction:
    """接受一个行为参数"""
    def __init__(self, statements, rewords, transformer, discount, actions):
        """
        :param statements:
                statements = ['A', 'B', ...]
        :param rewords:
                rewords = {
                    'A': 20,
                    'B': 60,
                    ...
                    }
        :param transformer:
                transformer = {
                        'A': {
                            'actionA' : {
                                'A' : 0.5
                                ...
                            }
                            'actionB' : {
                                'A' :0.2
                                ...
                            }
                            ...
                        },
                        'B': {
                            ....
                        },
                        ...
                    }
        :param discount:
        :param actions:
            ['action_one', 'action_two', ...]
        """
        self.statements = statements
        self.rewords = rewords
        self.transformer = transformer
        self.discount = discount
        self.actions = actions

    def compute_j(self, times):
        """计算前times的j """
        current_k = 1
        res = list()
        """res为map列表"""
        """初始化所有J(0) = 0 方便计算"""
        # res.append([{key: 0} for key in self.statements])
        init_map = {}
        for key in self.statements:
            init_map[key] = 0
        res.append(init_map)
        while current_k < times+1:
            temp_list = {}
            for statement in self.statements:
                current_j = 0
                # for the_neighbor_p in self.transformer[statement]:
                #     # current_k - 1即为上一步的J
                #     current_j += res[current_k - 1][the_neighbor_p] * self.transformer[statement][the_neighbor_p]
                # temp_list[statement] = self.rewords[statement] + current_j * self.discount
                j_with_max_action = -1000000000
                for action in self.actions:
                    j_with_action = 0
                    for the_neighbor in self.transformer[statement][action]:
                        j_with_action += res[current_k - 1][the_neighbor] * self.transformer[statement][action][the_neighbor]
                    j_with_action = self.rewords[statement] + j_with_action * self.discount
                    if j_with_action > j_with_max_action:
                        j_with_max_action = j_with_action
                temp_list[statement] = j_with_max_action
            res.append(temp_list)
            current_k += 1
        return res

    def store_as_file(self, file_name, res, round_number):
        """将res存入file_name的文件下"""
        times = 1
        if not os.path.exists(file_name):
            os.system(r"touch {}".format(file_name))  # 调用系统命令行来创建文件
        with open(file_name, 'w') as file_obj:
            for item in res[1:]:
                lines = ""
                for val in item.values():
                    lines += str(round(val, round_number)) + " "
                lines = "k = " + str(times) + " :" + lines + '\n'
                # print(lines)
                if times == 1:
                    title = ""
                    for statement in self.statements:
                        title += statement + "   "
                    title += '\n'
                    file_obj.writelines(title)
                file_obj.writelines(lines)
                times += 1




