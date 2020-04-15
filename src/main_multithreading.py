from src import ImageMoon
import numpy as np
import time
import threading
import pickle
from threading import Lock
threshold_low = 10000


class Score(object):
    def __init__(self, leng, nb_val):
        self.lock = Lock()
        self.value = np.zeros(leng)
        self.nb_valeurs = nb_val

    def increment(self, new_score):
        self.lock.acquire()
        try:
            self.value = self.value + new_score
            pass
        finally:
            self.lock.release()

    def set_value(self, arr):
        for i in range(len(arr)):
            self.value[i] = self.value[i] + arr[i]

    def store_results(self, title_test):
        best_score = 0
        best_index = 0
        for j in range(len(self.value)):
            if self.value[j] > best_score:
                best_score = self.value[j]
                best_index = j
        with open(f'results/scores_{title_test}', 'wb') as fp:
            pickle.dump(self.value, fp)
        with open(f'results/thresh_optimal_{title_test}', 'wb') as fp:
            result = {"threshold optimal": best_index, "score optimal": best_score}
            pickle.dump(result, fp)
        with open(f"results/readable_{title_test}.txt", "w") as fp:
            fp.write(f"best index :  {0.95 + best_index/ 1000}\nbest score : {best_score / self.nb_valeurs}\n")
            self.value.tofile(fp, "\n")


class UpdateScoreThread(threading.Thread):
    def __init__(self, score, image_number):
        super().__init__()
        self.image_number = image_number
        self.score = score

    def run(self):
        img = ImageMoon.ImageMoon(self.image_number)
        tmp_score = np.zeros(250)
        for j in range(0, 50):
            percent = 0.95 + j / 1000
            img.make_contours(percent, 0.1, threshold_low)
            true_pos, false_pos = img.compute_score()
            if true_pos or false_pos:
                tmp_score[j] = tmp_score[j] + (true_pos / (true_pos + false_pos))
        self.score.increment(tmp_score)


def best_threshold_all_rocks(ran):
    best_score = 0
    best_index = 0
    score = Score(50)
    start_time = time.gmtime().tm_sec
    for image_number in ran:
        proc = UpdateScoreThread(score, image_number)
        proc.start()
    for t in threading.enumerate():
        if t is not threading.currentThread():
            t.join()
    time_execution = str(time.gmtime().tm_sec - start_time)
    for j in range(len(score.value)):
        if score.value[j] > best_score:
            best_score = score.value[j]
            best_index = j


if __name__ == "__main__":
    best_threshold_all_rocks(range(500, 1000))
    pass
