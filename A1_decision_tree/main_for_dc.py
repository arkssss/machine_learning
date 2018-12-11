from dc_tree import DcTree

# 读取数据并初始化树
file_path = 'data.txt'
my_tree = DcTree(file_path)


# print(my_tree.TotalGain)
# print(my_tree.DataSet)
# print(my_tree.attr)
# print(my_tree.spilt_lists(my_tree.DataSet, 0, 'High'))

# print(my_tree.labels_value)
print(my_tree.build_tree())
