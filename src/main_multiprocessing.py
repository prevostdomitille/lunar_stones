import queue
import timeit
from multiprocessing import Process, Queue, Array, Lock
import numpy as np
import matplotlib.pyplot as plt
import pickle
from src import ImageMoon


def store_results(arr, title_test, min_img, max_img, param_range):
    """

    :param arr: The object to store
    :param title_test: "Name to identify the test"
    :param nb_images: Number of images for this test
    Dumps in two files the results for reuse and print in another
    file the results for monitoring
    """
    best_score = 0
    best_index = 0
    for j in range(len(arr[:])):
        if arr[j] > best_score:
            best_score = arr[j]
            best_index = j
    with open(f'results/scores_{title_test}', 'wb') as fp:
        pickle.dump(arr[:], fp)
    with open(f'results/thresh_optimal_{title_test}', 'wb') as fp:
        result = {"threshold optimal": best_index, "score optimal": best_score / (max_img - min_img)}
        pickle.dump(result, fp)
    with open(f"results/readable_{title_test}.txt", "w") as fp:
        fp.write(f"best index :  {param_range[best_index]}\nbest score : {best_score / (max_img - min_img) }\n")
        fp.write(f"Nombre d'images : {max_img - min_img}")


def run(img_number, scores_1, scores_2, params):
    """
    Computes the scores for parameters in the range provided by params and store it in
    objects scores_count and scores_jaccard
    """
    img = ImageMoon.ImageMoon(img_number)
    tmp_1 = np.zeros(len(params))
    tmp_2 = np.zeros(len(params))
    for j in range(len(params)):
        par = params[j]
        img.make_contours(0.99, par, 2000)
        score_any, score_small, score_big = img.scores_jaccard()
        tmp_1[j] = tmp_1[j] + score_small
        tmp_2[j] = tmp_2[j] + score_big
    for itr in range(len(params)):
        scores_1[itr] = scores_1[itr] + tmp_1[itr]
        scores_2[itr] = scores_2[itr] + tmp_2[itr]


def do_job(tasks_to_accomplish, scores_count, scores_jaccard, params):
    """
    Until there is no more image to treat in the queue of images, treat next image
    """
    while True:
        try:
            img_number = tasks_to_accomplish.get_nowait()
        except queue.Empty:
            break
        else:
            run(img_number, scores_count, scores_jaccard, params)


if __name__ == "__main__":
    lock = Lock()

    # Chose the images to work on
    min_img = 600
    max_img = 800
    nb_images = max_img - min_img
    # Chose the range of the parameter we want to evaluate
    param_range = np.linspace(0.05, 0.5, num=100)
    # the scores for parameter varying in param_range
    # they are multiprocessing objects
    scores_1 = Array('d', len(param_range), lock=lock)
    scores_2 = Array('d', len(param_range), lock=lock)

    # Create a list of images to treat
    tasks_to_accomplish = Queue()
    for i in range(min_img, max_img):
        tasks_to_accomplish.put(i)
    start = timeit.timeit()

    # To be set according to the number of kernels of the computer
    number_of_processes = 8
    processes = []

    # Start the processes
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish, scores_1, scores_2, param_range))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Store and plot results
    store_results(scores_1, "Jaccard", min_img, max_img, param_range)
    plt.plot(param_range, [i/nb_images for i in scores_1], 'r', label=f"Score de Jaccard pour les petites roches", )
    plt.plot(param_range, [i/nb_images for i in scores_2], label="Score de Jaccard pour les grandes roches")
    plt.title(label="Variation des scores de Jaccard en fonction du pourcentage de grandes images", fontsize=8)
    plt.xlabel("Proportion de grandes images")
    plt.ylabel(f"Scores moyens sur {nb_images} images")

    plt.legend()
    plt.show()

    # Measure execution time
    time_execution = timeit.timeit() - start
