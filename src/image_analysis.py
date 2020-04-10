import copy

from src import ImageMoon
import numpy as np
import threading

if __name__ == "__main__":
    # img = ImageMoon.ImageMoon(100)
    # img.make_contours(0.95, 0.1)
    # img.plot_results()
    # img.compute_score()
    best_score = 0
    scores = np.zeros(100)
    for i in range(1, 500):
        img = ImageMoon.ImageMoon(i)
        for j in range(0, 100):
            percent = 0.9 + j / 1000
            img.make_contours(percent, 0.1)
            true_pos, false_pos = img.compute_score()
            if true_pos or false_pos:
                scores[j] = scores[j] + (true_pos / (true_pos + false_pos))
    for j in range(len(scores)):
        if scores[j] > best_score:
            best_score = scores[j]
            best_value = j
    print(best_value)
    pass