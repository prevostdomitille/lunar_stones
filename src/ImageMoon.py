import copy

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tabulate

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
    length = int(len(list_copy) * percentage_big_rocks)
    criteria = 0
    if length:
        criteria = list_copy[-length]
    return criteria


def make_list_values_of_contours(contours, par=380):
    """

    :param contours: A list of contours detected on the image. Datatype : cv.contours
    :return: A list of values for the contours, computed as a function of size and distance
    """
    list_contours = []
    for cont in contours:
        bottommost = tuple(cont[cont[:, :, 1].argmin()][0])
        list_contours.append((475 - bottommost[1]) * (cv.contourArea(cont) + par))  # empirical
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


def plot_centers(image, centers, color=""):
    """
    :param color: default = white. Can be "green" for green dots or "red" for red dots.

    Draws the centers given as arguments on the image given as argument.
    Plots the figure
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


def compute_accuracy_positives(centers, truth):
    """
    Function used to compute the number of well positioned centered compared to
    ground truth image. A center is well positioned if it is somewhere in a rock
    of ground truth.

    :param centers: A list of coordinates
    :param truth: An image in black and white
    :returns: (nb_true, nb_false) : the number of centers positioned on a (light, black) area of the image

    """
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


def score(cont, truth, plot=False):
    """

    :param cont: contours
    :param truth: Ground truth
    :param plot: If true, plots the resulting image
    :return:
    """
    center = make_centers(cont)
    if plot:
        plot_centers(truth, center, color="green")
    return compute_accuracy_positives(center, truth)


def find_threshold_image(percentage, image):
    """

    :param percentage: The percentage of the ground that are rocks
    :param image: The image to be classified in black and white
    :return: Without considering the sky, returns the threshold above which the
    proper percentage of the image's pixels are situated.
    """
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


def score_jaccard(truth, pred):
    """

    :param truth: image black and white
    :param pred: image black and white
    :return: Le score de jaccard entre les deux images
    """
    intersec = cv.bitwise_and(truth, pred)
    union = cv.bitwise_or(truth, pred)
    if cv.countNonZero(union) == 0:
        return 0
    jaccard_score = cv.countNonZero(intersec) / cv.countNonZero(union)
    return jaccard_score


class ImageMoon:

    def __init__(self, img_number: int):
        self.number = img_number
        self.name_render = "render" + to_string(img_number) + ".png"
        self.name_ground = "ground" + to_string(img_number) + ".png"
        self.render = cv.imread(PATH + "render/" + self.name_render, 0)
        self.render_gray = cv.imread(PATH + "render/" + self.name_render, cv.COLOR_BGR2GRAY)
        self.render_blur = cv.medianBlur(self.render, 5)
        self.truth = cv.imread(PATH + "ground/" + self.name_ground, 1)
        self.prediction = None
        self.contours_small = []
        self.contours_big = []
        self.contours = []
        if len(self.render) == 0:
            raise ValueError(f"image {self.name_render} not found")

    def ground_gray(self):
        """Ground truth image in black and white without the sky"""
        sp = cv.split(self.truth)
        mask = sp[0] + sp[1]
        return mask

    def shape(self):
        return self.render.shape

    def truth_big_rocks(self):
        """Returns the ground truth for only big rocks in black and white"""
        b_img = cv.split(self.truth)
        _, img_big_rocks = cv.threshold(b_img[0], 127, 255, cv.THRESH_BINARY)
        return img_big_rocks

    def truth_small_rocks(self):
        """Returns the ground truth for only small rocks in black and white"""

        b_img = cv.split(self.truth)
        _, img_small_rocks = cv.threshold(b_img[1], 127, 255, cv.THRESH_BINARY)
        return img_small_rocks

    def plot_truth(self):
        """Plots the ground truth"""
        plotter_mult(self.truth, self.truth_small_rocks(), self.truth_big_rocks(),
                     title=["ground_truth", "Small rocks", "Big rocks"])

    def make_prediction_image(self):
        """

        Creates an image of predicted rocks
        """
        mask = np.zeros(self.shape(), np.uint8)
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        cv.drawContours(mask, self.contours_small, -1, (255, 0, 0), -1)
        cv.drawContours(mask, self.contours_big, -1, (0, 0, 255), -1)
        self.prediction = mask

    def filter_contours(self, nb_big_rocks, min_value):
        """

        :param nb_big_rocks: proportion of rocks that are big
        :param min_value: Value under which the contour is not considered a rock
        :return Classifies for the image the big contours / small contours and contours
        """
        contours_big = []
        contours_small = []
        list_values = make_list_values_of_contours(self.contours)
        criteria = find_criteria(list_values, nb_big_rocks)
        for i in range(len(self.contours)):
            hull = cv.convexHull(self.contours[i])
            if list_values[i] > criteria and list_values[i] > min_value:
                contours_big.append(hull)
            elif list_values[i] > min_value:
                contours_small.append(hull)
        self.contours_small = contours_small
        self.contours_big = contours_big
        self.contours = contours_small + contours_big

    def find_contours(self, thresh):
        """

        :param thresh: Threshold to use binary threshold
        :return: Use open CV to create a list of contours objects for this image
        """
        _, img = cv.threshold(self.render_blur, thresh, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.contours = contours

    def make_contours(self, percentage_luminosity=0.99, nb_big_rocks=0.1, min_value=6000):
        """Create a list of contours classified in big and small contours for this image"""
        thresh_luminosity = find_threshold_image(percentage_luminosity, self.render)
        self.find_contours(thresh_luminosity)
        self.filter_contours(nb_big_rocks, min_value)
        self.make_prediction_image()

    def score_false_positives_any(self):
        """

        :return: Number of well positioned contours divided by all the contours,
        regardless of size classification.
        Comprised between 0 and 1.
        """
        if self.prediction is None:
            self.make_contours()
        true_pos_any, false_pos_any = score(self.contours, self.ground_gray())
        if len(self.contours) == 0:
            return true_pos_any
        return true_pos_any / len(self.contours)

    def score_false_positives_classified(self):
        """Return a measure of the quality of the size classification :
        Well-classified_rocks / all well detected rocks
        Comprised between 0 and 1"""
        if self.prediction is None:
            self.make_contours()
        true_pos_big, _ = score(self.contours_big, self.truth_big_rocks())
        true_pos_small, _ = score(self.contours_small, self.truth_small_rocks())
        true_pos_any, _ = score(self.contours, self.ground_gray())
        if not true_pos_any:
            return 0
        return (true_pos_big + true_pos_small) / true_pos_any

    def scores_jaccard_classified(self):
        """Compute the number of well size classified pixels on all the well detected pixels"""
        if self.prediction is None:
            self.make_contours()
        pred = cv.split(self.prediction)
        all_good_pixels = cv.countNonZero(cv.bitwise_and(self.ground_gray(), pred[0] + pred[2]))
        good_smalls = cv.countNonZero(cv.bitwise_and(self.truth_small_rocks(), pred[0]))
        good_big = cv.countNonZero(cv.bitwise_and(self.truth_big_rocks(), pred[2]))
        if not all_good_pixels:
            print("error")
            return 0
        return (good_big + good_smalls) / all_good_pixels

    def scores_jaccard(self):
        """
        :return: Jaccard scores for all rocks, small rocks and big rocks
        """
        if self.prediction is None:
            self.make_contours()
        pred = cv.split(self.prediction)
        score_any = score_jaccard(self.ground_gray(), pred[0] + pred[2])
        score_small = score_jaccard(self.truth_small_rocks(), pred[0])
        score_big = score_jaccard(self.truth_big_rocks(), pred[2])

        return score_any, score_small, score_big

    def detailed_scores_false_positives(self, plot=False):
        """

        :param plot: If true, will plot the images of centers positioned on ground truth
        Prints a table with all the scores for this image
        """
        if not self.contours:
            return 0, 1
        true_pos_big, false_pos_big = score(self.contours_big, self.truth_big_rocks(), plot)
        true_pos_small, false_pos_small = score(self.contours_small, self.truth_small_rocks(), plot)
        true_pos_any, false_pos_any = score(self.contours, self.ground_gray(), plot)
        score_any, score_small, score_big = self.scores_jaccard()

        mydata = [("Petites roches", "|", true_pos_big, false_pos_big, score_small),
                  ("Grandes roches", "|", true_pos_small, false_pos_small, score_big),
                  ("Toutes roches", "|", true_pos_any, false_pos_any, score_any)]
        headers = ["Type", "", "Vrai positif", "Faux positifs", "Jaccard"]
        print(tabulate.tabulate(mydata, headers))

    def cumulated_histogram_dense(self):
        """

        :return: Histogram of the color shades cumulated without sky
        """
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


def validation():
    arr = []
    for i in range(1000, 1200):
        im = ImageMoon(i)
        im.make_contours(0.99, 0.2, 2000)
        arr.append(im.score_false_positives_any())
    plt.hist(arr, 20, cumulative=True, density=True)
    plt.xlabel("Score")
    plt.ylabel("Nombre d'images")
    plt.title("Histogramme des scores sur 200 images, thresh=0.99", fontsize=10)
    plt.show()


if __name__ == "__main__":
    validation()
