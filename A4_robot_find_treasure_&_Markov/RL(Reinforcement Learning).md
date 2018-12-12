## RL(Reinforcement Learning)

* 强化学习不同于监督学习, 监督学习的每一个数据集都会有一个正确标签, 从而可以拟合预测下一次数据集对应的标签. 强化学习的每次动作并没有对应的标签, 所以机器只有通过不断的尝试**从而获得这些数据集所对应的标签**, 因此强化学习比监督学习更近一步.



### 强化学习的方法分类

* 不理解环境 (Model - Free RL) : 没有模型, 只有从真实的环境中得到反馈从而学习

  **此种环境Agent只能按部就班, 等待显示世界中的反馈之后, 再做出下一步的Action**

  * Q-Learning 
  * Sarsa
  * Policy Gradients

* 理解环境(Model - Based RL) : 较之于 Model-Free , 这个方法就多了一个对于真实世界的建模的过程, 从而不仅可以从真实世界中反馈学习, 还可以再建模中进行反馈学习

  **此种环境中的Agent可以通过想象, 从而获得各种Action, 从中挑选一个最好的情况.**

  * Q-Learning 
  * Sarsa
  * Policy Gradients



####强化学习的决策基于

* 基于概率(Policy - Based RL)

  根据环境的反馈, 给出各个输出的Action的概率(虽然概率不同,但是都有可能被选中输出)

  * Policy Gradients

* 基于价值(Value - Based RL)

  根据环境的反馈, 给出各个输出的Action的价值(只有价值最高的Action才可以被选中输出)

  * Q - Learning
  * Sarsa


####强化学习的更新基于

* 回合更新 (Monte - Carlo Update)

  需要等待一回合游戏的结束, 才能开始总结这局游戏, 然后更新各个状态的值

  * 基础版 : Policy Gradients
  * Monte - Carlo Learning

* 单步更新 (Temporal - Difference Update)

  在游戏进行的每一步都在更新, 不用等待游戏的结束

  * Q - Learning
  * Sarsa
  * 升级版 : Policy Gradients



### Q - Learing

Q - Learning 是根据Q表进行决策的, Q表是一个 状态(S) | 行为(A) 的对应表. e.g.

|      | a1   | a2   |
| ---- | ---- | ---- |
| S1   | -2   | 1    |
| S2   | -4   | 2    |

此时在S1状态, 会从Q表中选择一个价值最高的行为, 此时i 为a2 , 作为下一个步骤的action

* 更新Q表: 由于Q-Learning为单步更新的算法, 所以在...之后, 进行更新Q表的操作, Q表的更新依赖于现实和估计两个部分
  * 估计部分为 : Q(S1, a2) 即当前Q表的真实值
  * 现实部分为 : R + $\gamma$*max(Q(s2))     [s2 为 s1的下一个状态], **将Q(s2) 的估计作为了Q(s1)的现实**

更新后的Q(s1, a2) = 之前那的Q(s1, a2) + $\alpha$ * gap (gap 为 现实 - 估计) 

* 具体的算法为 : 

  Initialize Q(s,a) arbitrarily	//  **一般初始化为全0**
  Repeat (for each episode):	//每一次学习 (即从起点到达终点的探索)
  ​	Initialize s 
  ​	Choose a from s using policy derived from Q (e.g, E-greedy)  //选择一个行为a ,  E-greedy为一个选择策略 
  ​	Take action a, observe r, s'    //**reword 不是Qtable的值**
  ​	Q(s,a) := Q(s, a) + $\alpha$ [R + $\gamma$*max(Q(s2))  - Q(s, a)]

  ​	 s := s'

  Unitl s in terminal 

* 具体需要实现以下几个函数: 

~~~python
#1 
def init_Q_table():
    """
    此时初始化一个Qtable, 
    默认的初始化状态为全 0 
    输入状态个数
    输入actiond的格式
    return 一个Qtable
    """
#2
def choose_action(state, q_table):
    """
    根据q_table 和 当前的状态 state
    选择出此时需要完成的动作 action (默认是选择val最大的action)
    如果q_table 的所有action的值相同 或者 随机到了E-greedy的随机选取部分, 则从action中随机选一个
    return 一个选择的动作 action
    """
#3
def get_env_feedback(S, A):
    """
    输入参数:
	S : 当前的状态
	A : 当前状态下所取的行为
    返回:
    S': 根据 S,A 计算出来的下一步的状态 //如果Reach the wall 则下一步状态就是本身
    R : 下一步状态的奖励 (一般除了终点(奖励>0), 其他时候奖励为0)(也可以加入惩罚, 为负数)
    """
    
#4 主循环函数
Q_table = init_Q_tabble(states, actions)
for i in range(max_episode):
    init_state s; # 设置初始化的状态 s
   	next_action = choose_action(s, Q_table) # 根据此时的状态s选择一个动作
    
    
    


~~~





















