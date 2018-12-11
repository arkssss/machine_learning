from Markov_Without_action import MarkovWithoutAction

statements = ['A.Assistant Prof', 'B.Assoc.Prof', 'T.Tenured Prof', 'S.On the Street', 'D. Dead']
rewords = {
    'A.Assistant Prof': 20,
    'B.Assoc.Prof': 60,
    'T.Tenured Prof': 400,
    'S.On the Street': 10,
    'D. Dead': 0
}
transformer = {
    'A.Assistant Prof': {
        'A.Assistant Prof': 0.6,
        'B.Assoc.Prof': 0.2,
        'T.Tenured Prof': 0,
        'S.On the Street': 0.2,
        'D. Dead': 0
    },
    'B.Assoc.Prof': {
        'A.Assistant Prof': 0,
        'B.Assoc.Prof': 0.6,
        'T.Tenured Prof': 0.2,
        'S.On the Street': 0.2,
        'D. Dead': 0
    },
    'T.Tenured Prof': {
        'A.Assistant Prof': 0,
        'B.Assoc.Prof': 0,
        'T.Tenured Prof': 0.7,
        'S.On the Street': 0,
        'D. Dead': 0.3
    },
    'S.On the Street': {
        'A.Assistant Prof': 0,
        'B.Assoc.Prof': 0,
        'T.Tenured Prof': 0,
        'S.On the Street': 0.7,
        'D. Dead': 0.3
    },
    'D. Dead': {
        'A.Assistant Prof': 0,
        'B.Assoc.Prof': 0,
        'T.Tenured Prof': 0,
        'S.On the Street': 0,
        'D. Dead': 1
    }
}
# 20 + 0.9*(12+12+2) = 20 + 26*0.9 =
discount = 0.9
Markov_one = MarkovWithoutAction(statements, rewords, transformer, discount)
# print(Markov_one.compute_j(30))
res = Markov_one.compute_j(30)
# for item in Markov_one.compute_j(30):
#     print(item.values())
file_name = "data_without_action.txt"
# 4为保留后四位
Markov_one.store_as_file(file_name, res, 10)


