import os


class TDAlg:

    def __init__(self, row_num, clo_num, reward):
        """初始化"""
        self.file_name = "Answer.txt"
        self.rows = row_num
        self.cols = clo_num
        # 回报map
        self.reward = reward
        self.map = dict()
        # 状态保存列表
        self.status = []
        self.Ns = dict()
        # 初始化图
        for i in range(1, self.rows+1):
            for j in range(1, self.cols+1):
                self.map[(i, j)] = 0.000000
                self.Ns[(i, j)] = 0

    def compute_path(self, pi):
        """根据pi计算新的map"""
        pre_status = []
        pre_reword = 0
        for current_status in pi:
            if current_status not in self.status:
                self.map[current_status] = self.reward[current_status]
                self.status.append(current_status)
            if not pre_status:
                # 第一次进来
                pre_status = current_status
                pre_reword = self.reward[current_status]
                self.print_map(current_status)
                continue
            else:
                self.Ns[(pre_status[0], pre_status[1])] += 1
                self.map[(pre_status[0], pre_status[1])] = self.map[(pre_status[0], pre_status[1])] + self.alpha(self.Ns[(pre_status[0], pre_status[1])]) * (pre_reword + self.map[current_status] - self.map[(pre_status[0], pre_status[1])])
            self.print_map(current_status)
            pre_status = current_status
            pre_reword = self.reward[current_status]

    @staticmethod
    def alpha(times):
        """计算alpha"""
        return float(60)/(59+times)

    @staticmethod
    def format_map(number):
        """将传入的值进行格式化"""
        ans = ""
        if number >= 0:
            ans += "+"
        if not number:
            ans += "0.000000"
        else:
            ans += str(number)
            index = ans.index('.')
            if len(ans) - index < 6:
                left_num = 6 - index
                the_zero = ""
                while left_num > 0:
                    the_zero += "0"
                    left_num -= 1
                ans += the_zero
            else:
                ans = str(round(float(ans), 6))
        ans += "   "
        return ans

    def print_map(self, pos):
        """二维数组打印出来"""
        print("Utilities after taking step " + "(" + str(pos[0]) + "," + str(pos[1]) + ")")
        self.save_data("Utilities after taking step " + "(" + str(pos[0]) + "," + str(pos[1]) + ")")
        col = self.cols
        row = self.rows
        index = 1
        while col > 0:
            ans = ""
            while index <= row:
                ans += self.format_map(self.map[(index, col)])
                index += 1
            self.save_data(ans)
            print(ans)
            index = 1
            col -= 1
        print()

    def save_data(self, the_str):
        """将数据保存到Answer.txt中"""
        if not os.path.exists(self.file_name):
            os.system(r"touch {}".format(self.file_name))  # 调用系统命令行来创建文件
        with open(self.file_name, 'a+') as file_obj:
            file_obj.writelines(the_str+"\n")

    # @staticmethod
    # def analysis_pi(pos, get_action):
    #     """
    #     pi为一个map的传入
    #     pi = {
    #     [1, 1]: "down"
    #     [1, 2]: "up"
    #     ...
    #     }
    #     pi((1,1))即为在(1,1)状态下一步的action
    #     解析pi的规则
    #     """
    #     if get_action == "down":
    #         pos[0] -= 1
    #     elif get_action == 'up':
    #         pos[0] += 1
    #     elif get_action == 'left':
    #         pos[1] -= 1
    #     elif get_action == 'right':
    #         pos[1] += 1
    #     else:
    #         pass
    #     return pos
