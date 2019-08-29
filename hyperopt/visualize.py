import numpy as np
import matplotlib.pyplot as plt


def vis_tuning_progress(cv_results, greater_is_better=False,
                        valid_score=None, n_initial_points=0):

    score = np.array(cv_results['mean_test_score'])
    iteration = np.array(range(1, len(score) + 1))

    if greater_is_better:
        best_score = np.maximum.accumulate(score)
    else:
        best_score = np.minimum.accumulate(score)

    if valid_score is not None:
        if greater_is_better:
            valid_mask = score > valid_score
        else:
            valid_mask = score < valid_score
        iteration = iteration[valid_mask]
        score = score[valid_mask]
        best_score = best_score[valid_mask]

    plt.plot(iteration, score, 'xkcd:silver')
    plt.plot(iteration, best_score, 'xkcd:goldenrod')

    if n_initial_points:
        plt.axvline(x=n_initial_points)

    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.show()
