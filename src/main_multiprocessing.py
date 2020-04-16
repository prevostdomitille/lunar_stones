import queue
import timeit
from multiprocessing import Process, Queue, Array, Lock
import numpy as np
import matplotlib.pyplot as plt
import pickle
from src import ImageMoon


def store_results(arr, title_test, nb_images):
    best_score = 0
    best_index = 0
    for j in range(len(arr[:])):
        if arr[j] > best_score:
            best_score = arr[j]
            best_index = j
    with open(f'results/scores_{title_test}', 'wb') as fp:
        pickle.dump(arr[:], fp)
    with open(f'results/thresh_optimal_{title_test}', 'wb') as fp:
        result = {"threshold optimal": best_index, "score optimal": best_score}
        pickle.dump(result, fp)
    with open(f"results/readable_{title_test}.txt", "w") as fp:
        fp.write(f"best index :  {0.95 + best_index / 1000}\nbest score : {best_score }\n")
        fp.write(f"Nombre d'images : {nb_images}")


def run(img_number, scores_count, scores_jaccard, t):
    img = ImageMoon.ImageMoon(img_number)
    tmp_score_count = np.zeros(len(t))
    tmp_jaccard = np.zeros(len(t))
    for j in range(len(t)):
        percent = t[j]
        img.make_contours(percent, 0.1, 7000)
        tmp_score_count[j] = tmp_score_count[j] + img.score_false_positives()
        tmp_jaccard[j] = tmp_jaccard[j] + img.score_jaccard()
    for itr in range(len(t)):
        scores_count[itr] = scores_count[itr] + tmp_score_count[itr]
        scores_jaccard[itr] = scores_jaccard[itr] + tmp_jaccard[itr]


def do_job(tasks_to_accomplish, scores_count, scores_jaccard, t):
    while True:
        try:
            img_number = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            run(img_number, scores_count, scores_jaccard, t)


if __name__ == "__main__":
    lock = Lock()
    number_of_processes = 8
    tasks_to_accomplish = Queue()
    processes = []
    # the images to make the statistics on
    min_img = 500
    max_img = 700
    ran = range(min_img, max_img)
    # the threshold is the percentage of the image that is a rock
    t = np.linspace(0.90, 0.990, num=100)
    scores_jac = Array('d', len(t), lock=lock)
    # the scores for a threshold varying from 0.95 to 1
    scores = Array('d', len(t), lock=lock)

    for i in ran:
        tasks_to_accomplish.put(i)
    start = timeit.timeit()

    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, scores, scores_jac, t))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    store_results(scores, "False negative optimisation", 300)
    plt.plot(t, [i / 30 for i in scores], label="Rock Count")
    plt.plot(t, scores_jac, label=f"Jaccard - {max_img - min_img}")
    plt.legend()
    plt.show()
    time_execution = timeit.timeit() - start
    print(f"time execution :{time_execution}")
