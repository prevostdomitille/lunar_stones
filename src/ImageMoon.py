import copy

import cv2 as cv
import numpy as np
import tabulate
import matplotlib.pyplot as plt
from src.help import to_string, plotter_mult, plotter

PATH = "/Users/domitilleprevost/Documents/PROJET_DEEP/artificial-lunar-rocky-landscape-dataset/images/"


def count_pixels(image, grey_level):
    return np.count_nonzero(image < grey_level)


def find_criteria(list_contours, percentage_big_rocks):
    """
    Given a list of items, return the value that
    represents a threshold under which a certain
    proportion of the values (= percentage_big_rockes)
    are situated.
    """
    list_copy = copy.deepcopy(list_contours)
    list_copy.sort()
    length = int(len(list_copy) * percentage_big_rocks)  # empirical
    criteria = 0
    if length:
        criteria = list_copy[-length]
    return criteria


def make_list_values_of_contours(contours):
    """

    :param contours: A list of contours detected on the image. Datatype : cv.contours
    :return: A list of values for the contours, computed as a function of size and distance
    """
    list_contours = []
    for cont in contours:
        bottommost = tuple(cont[cont[:, :, 1].argmin()][0])
        list_contours.append((470 - bottommost[1]) * (cv.contourArea(cont) + 10))  # empirical
    return list_contours


def make_centers(contours):
    """
    :param contours : A list of contours
    :return centers :  A list of tuples representing the centers of the contours (coordinates)
    """
    centers = []
    for cont in contours:
        M = cv.moments(cont)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
    return centers


def draw_centers(image, centers, color=""):
    """

    """

    im = copy.deepcopy(image)
    im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    for cent in centers:
        if color == "red":
            cv.circle(im, (cent[0], cent[1]), 5, (255, 0, 0), -1)
        elif color == "green":
            cv.circle(im, (cent[0], cent[1]), 5, (0, 255, 0), -1)
        else:
            cv.circle(im, (cent[0], cent[1]), 5, (255, 255, 255), -1)
    plotter(im)
    return centers


def compute_accuracy_positives(centers, truth):
    nb_true = 0
    nb_false = 0
    for cent in centers:
        try:
            if truth[cent[1], cent[0]] > 10:
                nb_true += 1
            else:
                nb_false += 1
        except IndexError:
            print("warning with index value")
    return nb_true, nb_false


def find_threshold_no_sky(percentage, image):
    j = 10
    thresh_value = 0
    nb_pixels = (count_pixels(image, 255) - count_pixels(image, 10))
    while j < 255:
        cur = (count_pixels(image, (j + 1)) - count_pixels(image,
                                                                 10)) / nb_pixels  # - count_pixels(image, j * 10)
        if cur > percentage and thresh_value == 0:
            thresh_value = j + 1
        j = j + 1
    return thresh_value


class ImageMoon:

    def __init__(self, img_number: int):
        self.number = img_number
        self.name_render = "render" + to_string(img_number) + ".png"
        self.name_ground = "ground" + to_string(img_number) + ".png"
        self.render = cv.imread(PATH + "render/" + self.name_render, 0)
        self.render_gray = cv.imread(PATH + "render/" + self.name_render, cv.COLOR_BGR2GRAY)
        self.ground = cv.imread(PATH + "ground/" + self.name_ground, 1)
        self.prediction = np.zeros(self.render_gray.shape)
        self.contours_small = []
        self.contours_big = []
        self.contours = []
        self.render_blur = cv.medianBlur(self.render, 5)
        if len(self.render) == 0:
            raise ValueError(f"image {self.name_render} not found")

    def ground_gray(self):
        sp = cv.split(self.ground)
        mask = sp[0] + sp[1]
        return mask

    def shape(self):
        return self.render.shape

    def truth_big_rocks(self):
        b_img = cv.split(self.ground)
        _, img_big_rocks = cv.threshold(b_img[0], 127, 255, cv.THRESH_BINARY)
        return img_big_rocks

    def truth_small_rocks(self):
        b_img = cv.split(self.ground)
        _, img_small_rocks = cv.threshold(b_img[1], 127, 255, cv.THRESH_BINARY)
        return img_small_rocks

    def disp_truth(self):
        plotter_mult(self.ground, self.truth_small_rocks(), self.truth_big_rocks(),
                     title=["ground_truth", "Small rocks", "Big rocks"])

    def prediction_image(self):
        mask = np.zeros(self.shape(), np.uint8)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        cv.drawContours(mask, self.contours_small, -1, (255, 0, 0), -1)
        cv.drawContours(mask, self.contours_big, -1, (0, 0, 255), -1)
        self.prediction = mask

    def filter_contours(self, contours, nb_big_rocks, min_value):
        contours_big = []
        contours_small = []
        list_values = make_list_values_of_contours(contours)
        criteria = find_criteria(list_values, nb_big_rocks)
        for i in range(len(contours)):
            hull = cv.convexHull(contours[i])
            if list_values[i] > criteria and list_values[i] > min_value:
                contours_big.append(hull)
            elif list_values[i] > min_value:
                contours_small.append(hull)
        self.contours_small = contours_small
        self.contours_big = contours_big
        self.contours = contours_small + contours_big
        return contours_small, contours_big

    def find_contours(self, thresh):
        ret, img = cv.threshold(self.render_blur, thresh, 255, cv.THRESH_BINARY)
        contours, image = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.contours = contours

    def make_contours(self, percentage_luminosity, nb_big_rocks, min_value):
        thresh_luminosity = find_threshold_no_sky(percentage_luminosity, self.render)
        self.find_contours(thresh_luminosity)
        self.filter_contours(self.contours, nb_big_rocks, min_value)

    def score_false_negatives(self):
        self.prediction_image()
        nb_pix = cv.countNonZero(self.ground_gray())
        nb_predicted = cv.countNonZero(cv.cvtColor(self.prediction, cv.COLOR_BGR2GRAY))
        return abs(nb_pix - nb_predicted) / nb_pix

    def score_false_positives(self):
        centers = make_centers(self.contours)
        true_pos_any, false_pos_any = compute_accuracy_positives(centers, self.ground_gray())
        if len(self.contours) == 0:
            return true_pos_any
        return true_pos_any / len(self.contours)

    def score_jaccard(self):
        self.prediction_image()
        truth = self.ground_gray()
        pred = cv.cvtColor(self.prediction, cv.COLOR_BGR2GRAY)
        intersec = cv.bitwise_and(truth, pred)
        union = cv.bitwise_or(truth, pred)
        if cv.countNonZero(union) == 0:
            return 0
        jaccard_score = cv.countNonZero(intersec) / cv.countNonZero(union)
        return jaccard_score

    def detailed_scores_false_positives(self):
        if not self.contours_big:
            self.make_contours(0.99, 0.1, 8000)

        centers_big = make_centers(self.contours_big)
        centers_small = make_centers(self.contours_small)
        centers = make_centers(self.contours)

        draw_centers(self.truth_big_rocks(), centers_big, color="green")
        draw_centers(self.truth_small_rocks(), centers_small, color="green")
        true_pos_big, false_pos_big = compute_accuracy_positives(centers_small, self.truth_small_rocks())
        true_pos_small, false_pos_small = compute_accuracy_positives(centers_big, self.truth_big_rocks())
        true_pos_any, false_pos_any = compute_accuracy_positives(centers, self.ground_gray())

        mydata = [("Petites roches", "|", true_pos_big, false_pos_big),
                  ("Grandes roches", "|", true_pos_small, false_pos_small),
                  ("Toutes roches", "|", true_pos_any, false_pos_any)]
        headers = ["Type", "", "Vrai positif", "Faux positifs"]
        print(tabulate.tabulate(mydata, headers))
        return true_pos_any, false_pos_any

    def cumulated_histogram_dense(self):
        t = np.zeros(245)
        x = np.arange(10, 255)
        j = 10
        nb_pixels = (count_pixels(self.render, 255) - count_pixels(self.render, 10))
        while j < 255:
            t[j - 10] = (count_pixels(self.render, (j + 1)) - count_pixels(self.render,
                                                                           10)) / nb_pixels  # - count_pixels(image,
            # j * 10)
            j = j + 1
        plt.title("Histogramme dense sans le ciel")
        plt.legend()
        plt.plot(x, t, label=f"{self.shape}")
        plt.show()
        return t


if __name__ == "__main__":
    im = ImageMoon(2)
    for i in range(300, 500):
        im = ImageMoon(i)
        im.make_contours(0.97, 0.1, 1000)
        im.score_jaccard()

    im.make_contours(0.97, 0.1, 1000)
    print(im.score_false_negatives())
    im.score_jaccard()
    print(im.shape()[0])
