from math import log
import os

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression


def indicator_calculator(correct_file, train_data):
    Y = []
    answers = {}
    times = {}
    correct_answers = []

    with open(correct_file) as data_file:
        line = data_file.read().strip("\n")
        correct_answers = [int(i) for i in line.split(",")]

    with open(train_data) as data_file:
        counter = 0
        while True:
            line = data_file.readline().strip("\n")
            if not line:
                break
            Y.append(int(line))

            line = data_file.readline().strip("\n")
            if not line:
                break
            line = [int(i) for i in line.split(",")]
            answers[counter] = {}
            for question, user_answer in enumerate(line):
                answers[counter][question] = (
                    1 if user_answer == correct_answers[question] else 0
                )

            line = data_file.readline().strip("\n")
            if not line:
                break
            line = [int(i) for i in line.split(",")]
            times[counter] = {}
            for question, user_time in enumerate(line):
                before_time = line[question - 1] if question - 1 > 0 else 0
                times[counter][question] = user_time - before_time

            counter += 1

    g = list(range(0, len(correct_answers)))
    xn = list(answers.keys())
    S_Xn = {}
    for i in xn:
        S_Xn[i] = sum(answers[i].values())

    P_g = {}
    for i in g:
        sum_xng = 0
        for j in answers:
            sum_xng += answers[j][i]
        P_g[i] = sum_xng / len(xn)

    W_g = {}
    for i in g:
        W_g[i] = log(P_g[i] / (1 - P_g[i]))

    def get_u3(user_id):
        if not (0 < S_Xn[user_id] < len(g)):
            return 0

        first_def = 0
        for i in range(0, S_Xn[user_id]):
            first_def += W_g[i]

        second_def = 0
        for i in g:
            second_def += answers[user_id][i] * W_g[i]

        third_def = 0
        for i in range(len(g) - S_Xn[user_id] + 1, len(g)):
            third_def += W_g[i]

        if first_def == 0 or third_def == 0 or second_def == 0:
            return 0
        return (first_def - second_def) / (first_def - third_def)

    avg_t_g = {}

    for i in g:
        avg_t_g[i] = sum([times[j][i] for j in xn])

    for i in avg_t_g:
        avg_t_g[i] = avg_t_g[i] / len(times)

    F_Tng = {}
    for i in times:
        total = sum(list(times[i].values()))
        F_Tng[i] = {}
        for j in times[i]:
            F_Tng[i][j] = times[i][j] / total

    F_avg_Tg = {}
    for i in g:
        F_avg_Tg[i] = 0
        for j in times:
            F_avg_Tg[i] += times[j][i]

    total_avg = sum(list(F_avg_Tg.values()))
    for i in g:
        F_avg_Tg[i] = F_avg_Tg[i] / total_avg

    def get_kl(user_id):
        result = 0
        for i in g:
            result += F_avg_Tg[i] * log(F_avg_Tg[i] / F_Tng[user_id][i])
        return result

    X = []
    for i in xn:
        X.append([get_u3(i), get_kl(i)])
    return X, Y


DIR_PATH = os.path.abspath(os.path.dirname(__file__))

TRAIN_X, TRAIN_Y = indicator_calculator(
    DIR_PATH + "/data/correct_answers1.txt", DIR_PATH + "/data/raw_data1.txt"
)
TEST_X, TEST_Y = indicator_calculator(
    DIR_PATH + "/data/correct_answers2.txt", DIR_PATH + "/data/raw_data2.txt"
)

TRAIN_X = preprocessing.StandardScaler().fit(TRAIN_X).transform(TRAIN_X)
TEST_X = preprocessing.StandardScaler().fit(TEST_X).transform(TEST_X)

clf = LogisticRegression(random_state=0).fit(TRAIN_X, TRAIN_Y)
Y_PRED = clf.predict(TEST_X)
print(precision_score(TEST_Y, Y_PRED, average=None))
