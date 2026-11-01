from vllm.v1.worker.cold_start_transition import compute_migration_plan


def test_nonadjacent_descending_migration():
    plan = compute_migration_plan([(0, 1), (1, 2), (2, 3), (3, 16)],
                                  [(0, 4), (4, 8), (8, 12), (12, 16)])
    assert plan[3]['send'] == {0: [3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11]}
    assert plan[0]['recv'] == {1: [1], 2: [2], 3: [3]}


def test_nonadjacent_ascending_migration():
    plan = compute_migration_plan([(0, 13), (13, 14), (14, 15), (15, 16)],
                                  [(0, 4), (4, 8), (8, 12), (12, 16)])
    assert plan[0]['send'] == {1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12]}
    assert plan[3]['recv'] == {0: [12], 1: [13], 2: [14]}
