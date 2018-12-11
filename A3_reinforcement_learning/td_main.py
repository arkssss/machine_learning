from td_alg import TDAlg

the_map = {(1, 2): 1}


reword = {

    (1, 1): -0.04,
    (1, 2): -0.04,
    (1, 3): -0.04,
    (2, 1): -0.04,
    (2, 2): -0.04,
    (2, 3): -0.04,
    (3, 1): -0.04,
    (3, 2): -0.04,
    (3, 3): -0.04,
    (4, 1): -0.04,
    (4, 2): -1.00,
    (4, 3): 1.00

}
the_td = TDAlg(4, 3, reword)
the_td.save_data("Trial 1:")
pi1 = [(1, 1), (1, 2), (1, 3), (1, 2), (1, 3), (2, 3), (3, 3), (4, 3)]
the_td.compute_path(pi1)

the_td.save_data("Trial 2:")
pi2 = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (2, 3), (3, 3), (4, 3)]
the_td.compute_path(pi2)

the_td.save_data("Trial 3:")
pi3 = [(1, 1), (2, 1), (3, 1), (3, 2), (4, 2)]
the_td.compute_path(pi3)


