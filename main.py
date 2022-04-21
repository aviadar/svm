from sklearn import datasets, svm as sk_svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# for Kernel Trick
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# for Kernel Trick
def polynomial_kernel(x1, x2, p=3):
    return (1 + np.dot(x1, x2)) ** p


class SVM:
    image_counter = 0  # used to save plots

    def __init__(self, kernel=linear_kernel, C=1.0, tolerance=0.001, max_passes=10, random_state=None,
                 visualization=False, plot_plane=True):
        self.kernel = kernel  # the default uses no kernel (linear kernel)
        self.C = C  # allows margin (C=1) to margins are heavily fined (C=100)
        self.tolerance = tolerance  # tolernace for convergence
        self.max_passes = max_passes  # also a tolernace for convergence
        self.alphas = None  # aka lambdas - as stated in the dual problem
        self.kernel_results = None
        self.random_state = random_state
        self.visualization = visualization  # flag for plotting
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

        self.plot_plane = plot_plane

    def calc_kernel(self, X):
        self.kernel_results = np.zeros((X.shape[0], X.shape[0]))

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                self.kernel_results[i, j] = self.kernel(X[i], X[j])

    def calc_f(self, x, b):
        f = 0.0
        for i in range(self.X.shape[0]):
            f += self.alphas[i] * self.y[i] * self.kernel(self.X[i, :], x)
        f += b
        return f

    @staticmethod
    # as part of the smo algorithm - alphaj(lambda_j) is selected randomly
    def choose_j_randomly(i, n_samples, random_state):
        j = i
        while j == i:
            j = random_state.randint(0, n_samples - 1)
        return j

    # calc higher and lower bounds on alphaj
    def calc_L_H(self, yi, yj, alphai, alphaj):
        if yi == yj:
            L = max(0.0, alphaj + alphai - self.C)
            H = min(self.C, alphaj + alphai)
        else:
            L = max(0.0, alphaj - alphai)
            H = min(self.C, self.C + alphaj - alphai)

        return L, H

    def calc_eta(self, xi, xj):
        return 2 * self.kernel(xi, xj) - self.kernel(xi, xi) - self.kernel(xj, xj)

    # separating hyper plane calculation
    def calc_w_star(self):
        return sum([self.alphas[i] * self.y[i] * self.X[i, :] for i in range(self.X.shape[0])])

    # fitting is done according to SMO
    def fit(self, X, y):
        self.X = X
        self.y = y

        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        b = 0.0
        passes = 0
        it = 0

        while passes <= self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                f = self.calc_f(X[i, :], b)
                Ei = f - y[i]
                if (y[i] * Ei < -self.tolerance and self.alphas[i] < self.C) or (
                        y[i] * Ei > self.tolerance and self.alphas[i] > 0.0):
                    j = SVM.choose_j_randomly(i, n_samples, self.random_state)
                    f = self.calc_f(X[j, :], b)
                    Ej = f - y[j]
                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]
                    L, H = self.calc_L_H(y[i], y[j], self.alphas[i], self.alphas[j])
                    if abs(L - H) < 1e-4:
                        continue
                    eta = self.calc_eta(X[i, :], X[j, :])
                    if eta >= 0:
                        continue

                    # calc alpha_j
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L

                    if abs(self.alphas[j] - alpha_j_old) < 1e-4:
                        continue

                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    b1 = b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i, :], X[i, :]) - y[j] * (
                            self.alphas[j] - alpha_j_old) * self.kernel(X[i, :], X[j, :])
                    b2 = b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i, :], X[j, :]) - y[j] * (
                            self.alphas[j] - alpha_j_old) * self.kernel(X[j, :], X[j, :])

                    if 0 < self.alphas[i] < self.C:
                        b = b1
                    elif 0 < self.alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2

                    num_changed_alphas += 1
            print(f'{it=}:{self.alphas=}')
            it += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

            if self.visualization:
                self.w_star = self.calc_w_star()
                self.b_star = b
                self.visualize()

        self.w_star = self.calc_w_star()
        self.b_star = b

        if self.visualization:
            self.visualize(show=True)

    # label prediction
    def predict(self, x):
        return np.sign(self.b_star + sum(
            [self.alphas[i] * self.y[i] * self.kernel(self.X[i, :], x) for i in range(self.X.shape[0])]))

    # plotting
    def plot_2d_data(self):
        target_plus_indicies = np.where(self.y == 1)[0]
        target_minus_indicies = np.where(self.y == -1)[0]
        sv_indicies = np.where(self.alphas > 0)[0]
        self.ax.plot(self.X[target_plus_indicies, 0], self.X[target_plus_indicies, 1], 'o', label=1)
        self.ax.plot(self.X[target_minus_indicies, 0], self.X[target_minus_indicies, 1], 'o', label=-1)
        self.ax.plot(self.X[sv_indicies, 0], self.X[sv_indicies, 1], 'o', markersize=14, markerfacecolor="None",
                     label='sv')
        plt.legend()
        self.ax.set_xlabel('x1')
        self.ax.set_ylabel('x2')
        self.ax.axis('equal')
        # plt.show()

    # plotting
    def visualize(self, show=False):
        self.ax.cla()

        self.plot_2d_data()
        min_feature_value = np.min(self.X)
        max_feature_value = np.max(self.X)
        self.ax.set_xlim(min_feature_value, max_feature_value)
        self.ax.set_ylim(min_feature_value, max_feature_value)

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (min_feature_value * 0.9, max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        if self.plot_plane:
            # (w.x+b) = 1
            # positive support vector hyperplane
            psv1 = hyperplane(hyp_x_min, self.w_star, self.b_star, 1)
            psv2 = hyperplane(hyp_x_max, self.w_star, self.b_star, 1)
            self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

            # (w.x+b) = -1
            # negative support vector hyperplane
            nsv1 = hyperplane(hyp_x_min, self.w_star, self.b_star, -1)
            nsv2 = hyperplane(hyp_x_max, self.w_star, self.b_star, -1)
            self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

            # (w.x+b) = 0
            # positive support vector hyperplane
            db1 = hyperplane(hyp_x_min, self.w_star, self.b_star, 0)
            db2 = hyperplane(hyp_x_max, self.w_star, self.b_star, 0)
            self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        self.plot_contours(cmap=plt.cm.coolwarm, alpha=0.8)

        precounter = ''
        for i in range(3 - len(str(SVM.image_counter))):
            precounter += '0'

        plt.title('iteration ' + str(SVM.image_counter))
        if not show:
            plt.savefig('image_' + precounter + str(SVM.image_counter) + '.png')
            SVM.image_counter += 1
            plt.pause(0.01)
        else:
            plt.savefig('image_' + precounter + str(SVM.image_counter) + '.png')
            SVM.image_counter += 1
            plt.show()

    # plotting
    def make_meshgrid(self, h=0.02):
        x_min, x_max = np.min(self.X[:, 0]) - 1.5, np.max(self.X[:, 0]) + 1.5
        y_min, y_max = np.min(self.X[:, 1]) - 1.5, np.max(self.X[:, 1]) + 1.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    # plotting
    def plot_contours(self, **params):
        xx, yy = self.make_meshgrid()
        Z = np.array([self.predict([xx_i, yy_i]) for xx_i, yy_i in zip(xx, yy)])
        # Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = self.ax.contourf(xx, yy, Z, **params)
        return out


## EXAMPLES PART ##

def example_1():
    # EXAMPLE 1
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 2)
    b = -0.2
    w = np.array([0.5, -0.3])
    y = np.sign(b + np.dot(X, w))

    svm = SVM(C=100.0, random_state=random_state, visualization=True)
    svm.fit(X, y)


def example_1_1():
    # EXAMPLE 1_1
    random_state = np.random.RandomState(0)
    X = random_state.randn(10, 2)
    b = -0.2
    w = np.array([0.5, -0.3])
    y = np.sign(b + np.dot(X, w))
    y[3] = -y[3]

    svm = SVM(C=100.0, random_state=random_state, visualization=True)
    svm.fit(X, y)


def example_2():
    ## EXAMPLE 2
    iris = datasets.load_iris()
    X = iris['data'][:, (2, 3)]
    scaler = StandardScaler()
    Xstan = scaler.fit_transform(X)

    data = pd.DataFrame(data=Xstan, columns=['petal length', 'petal width'])
    data['target'] = iris['target']
    data = data[data['target'] != 2]  # we will only focus on Iris-setosa and Iris-Versicolor
    data['target'].iloc[data['target'] == 0] = -1
    # data = data.loc[(0,1,2,3,4,90,91,92,93,94),:]

    random_state = np.random.RandomState(0)
    svm = SVM(C=100.0, random_state=random_state, visualization=True)
    svm.fit(data.loc[:, ['petal length', 'petal width']].values, data['target'].values)


def example_2_1():
    ## EXAMPLE 2_1
    iris = datasets.load_iris()
    X = iris['data'][:, (2, 3)]
    scaler = StandardScaler()
    Xstan = scaler.fit_transform(X)

    data = pd.DataFrame(data=Xstan, columns=['petal length', 'petal width'])
    data['target'] = iris['target']
    data = data[data['target'] != 2]  # we will only focus on Iris-setosa and Iris-Versicolor
    data['target'].iloc[data['target'] == 0] = -1
    # data = data.loc[(0,1,2,3,4,90,91,92,93,94),:]
    data['target'].iloc[20] = -data['target'].iloc[20]

    random_state = np.random.RandomState(0)
    svm = SVM(C=100.0, random_state=random_state, visualization=True)
    svm.fit(data.loc[:, ['petal length', 'petal width']].values, data['target'].values)


def example_2_2():
    ## EXAMPLE 2_2
    iris = datasets.load_iris()
    X = iris['data'][:, (2, 3)]
    scaler = StandardScaler()
    Xstan = scaler.fit_transform(X)

    data = pd.DataFrame(data=Xstan, columns=['petal length', 'petal width'])
    data['target'] = iris['target']
    data = data[data['target'] != 2]  # we will only focus on Iris-setosa and Iris-Versicolor
    data['target'].iloc[data['target'] == 0] = -1
    # data = data.loc[(0,1,2,3,4,90,91,92,93,94),:]
    data['target'].iloc[20] = -data['target'].iloc[20]

    random_state = np.random.RandomState(0)
    svm = SVM(C=1.0, random_state=random_state, visualization=True)
    svm.fit(data.loc[:, ['petal length', 'petal width']].values, data['target'].values)


def example_3():
    ## EXAMPLE 3
    random_state = np.random.RandomState(0)
    X, y = datasets.make_moons(noise=0.1, random_state=2)
    y[y == 0] = -1
    svm = SVM(C=1.0, random_state=random_state, visualization=True)
    svm.fit(X, y)


def example_4():
    ## EXAMPLE 4
    random_state = np.random.RandomState(0)
    X, y = datasets.make_moons(noise=0.1, random_state=2)
    y[y == 0] = -1
    svm = SVM(C=1.0, kernel=polynomial_kernel, random_state=random_state, visualization=True, plot_plane=False)
    svm.fit(X, y)


def example_5():
    ## EXAMPLE 5
    # download fer2013.csv form https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge and rename it to train.csv
    df = pd.read_csv('train.csv')
    df1 = df[np.logical_or(df['emotion'] == 3, df['emotion'] == 4)]
    img_array = df1.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48 * 48).astype('float32'))

    X = np.stack(img_array, axis=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # consider only the bottom half
    # X = X[:, range(int(48*48/2), X.shape[1])]
    # X = X[:, range(int(48 * 48 / 2))]

    y = df1['emotion'].values
    y[y == 3] = -1
    y[y == 4] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # X = img_array[range(100),:]
    # y = df1['emotion'].values[range(100)]
    X_train = X_train[range(500), :]
    y_train = y_train[range(500)]
    X_test = X_test[range(300), :]
    y_test = y_test[range(300)]
    random_state = np.random.RandomState(0)
    svm = SVM(C=1.0, kernel=polynomial_kernel, random_state=random_state, visualization=False)
    svm.fit(X_train, y_train)

    # accuracy = 0
    # for x, actual_y in zip(X_test, y_test):
    #     pred_y = svm.predict(x)
    #     if pred_y == actual_y:
    #         accuracy += 1
    #
    # accuracy /= X_test.shape[0]
    # print(f'{accuracy=}')
    accuracy = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    for x, actual_y in zip(X_test, y_test):
        pred_y = svm.predict(x)
        if pred_y == actual_y:
            accuracy += 1
            if pred_y == 1:
                true_positive += 1
        else:
            if pred_y == 1:
                false_positive += 1
            else:
                false_negative += 1

    print('OUR SVM, using SMO:')
    accuracy /= X_test.shape[0]
    print(f'{accuracy=}')
    precision = true_positive / (true_positive + false_positive)
    print(f'{precision=}')
    recall = true_positive / (true_positive + false_negative)
    print(f'{recall=}')
    f1 = 2 * precision * recall / (precision + recall)
    print(f'{f1=}')
    print('')


def example_6():
    ## EXAMPLE 6
    # download data.csv form https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
    df = pd.read_csv('data.csv')
    X = df[['radius_mean', 'texture_mean', 'perimeter_mean',
            'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
            'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
            'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
            'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
            'fractal_dimension_se', 'radius_worst', 'texture_worst',
            'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst',
            'symmetry_worst', 'fractal_dimension_worst']].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = df['diagnosis'].values
    y[y == 'M'] = -1
    y[y == 'B'] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    random_state = np.random.RandomState(0)
    svm = SVM(C=1.0, kernel=polynomial_kernel, random_state=random_state, visualization=False)
    svm.fit(X_train, y_train)

    accuracy = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    for x, actual_y in zip(X_test, y_test):
        pred_y = svm.predict(x)
        if pred_y == actual_y:
            accuracy += 1
            if pred_y == 1:
                true_positive += 1
        else:
            if pred_y == 1:
                false_positive += 1
            else:
                false_negative += 1

    print('OUR SVM, using SMO:')
    accuracy /= X_test.shape[0]
    print(f'{accuracy=}')
    precision = true_positive / (true_positive + false_positive)
    print(f'{precision=}')
    recall = true_positive / (true_positive + false_negative)
    print(f'{recall=}')
    f1 = 2 * precision * recall / (precision + recall)
    print(f'{f1=}')
    print('')

    y_train = y_train.astype('int')
    clf = sk_svm.SVC(C=1.0, kernel='poly', random_state=42).fit(X_train, y_train)

    accuracy = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0
    for x, actual_y in zip(X_test, y_test):
        pred_y = clf.predict(x.reshape(1, -1))
        if pred_y == actual_y:
            accuracy += 1
            if pred_y == 1:
                true_positive += 1
        else:
            if pred_y == 1:
                false_positive += 1
            else:
                false_negative += 1

    print('sklearn:')
    accuracy /= X_test.shape[0]
    print(f'{accuracy=}')
    precision = true_positive / (true_positive + false_positive)
    print(f'{precision=}')
    recall = true_positive / (true_positive + false_negative)
    print(f'{recall=}')
    f1 = 2 * precision * recall / (precision + recall)
    print(f'{f1=}')


example_6()
