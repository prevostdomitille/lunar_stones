import queue
import timeit
from multiprocessing import Process, Queue, Array, Lock
import numpy as np
import pickle
from src import ImageMoon


def store_results(arr, title_test):
    best_score = 0
    best_index = 0
    for j in range(len(arr)):
        if arr[j] > best_score:
            best_score = [j]
            best_index = j
    with open(f'results/scores_{title_test}', 'wb') as fp:
        pickle.dump(arr, fp)
    with open(f'results/thresh_optimal_{title_test}', 'wb') as fp:
        result = {"threshold optimal": best_index, "score optimal": best_score}
        pickle.dump(result, fp)
    with open(f"results/readable_{title_test}.txt", "w") as fp:
        fp.write(f"best index :  {0.95 + best_index / 1000}\nbest score : {best_score }\n")
        arr.tofile(fp, "\n")


def run(img_number, scores):
    img = ImageMoon.ImageMoon(img_number)
    tmp_score = np.zeros(len(scores))
    for j in range(0, 50):
        percent = 0.95 + j / 1000
        img.make_contours(percent, 0.1, 10000)
        true_pos, false_pos = img.compute_score()
        if true_pos or false_pos:
            tmp_score[j] = tmp_score[j] + (true_pos / (true_pos + false_pos))
    for itr in range(len(scores)):
        scores[itr] = scores[itr] + tmp_score[itr]


def do_job(tasks_to_accomplish, scores):
    while True:
        try:
            img_number = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            run(img_number, scores)


if __name__ == "__main__":
    lock = Lock()
    number_of_processes = 8
    nb_images = 500
    tasks_to_accomplish = Queue()
    processes = []
    scores = Array('d', 50, lock=lock)

    for i in range(nb_images):
        tasks_to_accomplish.put(i + 300)

    start = timeit.timeit()

    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, scores))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    store_results(ob, "multiprocessing")
    time_execution = timeit.timeit() - start
    print(f"time execution :{time_execution}")
    print(ob[:])
