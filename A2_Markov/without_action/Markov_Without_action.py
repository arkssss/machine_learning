import os
class MarkovWithoutAction:

    def __init__(self, statements, rewords, transformer, discount):
        """假设有n个状态， statements = 1*n rewords = 1*n transformer = n*n"""
        """reword为map"""
        """{
            'statement_one' : reword
            ...
        }"""
        """transformer转移概率为二维map
        {
        'statement_one' :{
            'statement_one' : p1,
            'statement_two' : p2
            ...
        }
        ...
        }
        """
        self.statements = statements
        self.statements_num = len(self.statements)
        self.rewords = rewords
        self.transformer = transformer
        # 每年衰减
        self.discount = discount

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
                for the_neighbor_p in self.transformer[statement]:
                    # current_k - 1即为上一步的J
                    current_j += res[current_k - 1][the_neighbor_p] * self.transformer[statement][the_neighbor_p]
                temp_list[statement] = self.rewords[statement] + current_j * self.discount
            res.append(temp_list)
            current_k += 1
        return res

    def store_as_file(self, file_name, res, round_number):
        """将res存入file_name的文件下, round_number为保留小数点后的位数"""
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





