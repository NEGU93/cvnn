
if __name__ == "__main__":
    # monte_carlo_loss_gaussian_noise(iterations=100, filename="historgram_gaussian.csv")
    m = 100000
    n = 100
    num_classes = 4
    name = 'hilbert'
    # random
    # vect = None
    # work
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # vect = [42, 51, 37]
    # Doesn't work
    # vect = [42, 51, 37]

    """
    x_train, y_train, x_test, y_test = dp.get_gaussian_noise(m, n, num_classes, name)
    # x_train, y_train, x_test, y_test = dp.get_constant(m, n, num_classes, vect)
    x_train_real, x_test_real = dp.get_real_train_and_test(x_train, x_test)

    cvnn, rvnn = do_one_iter(x_train, y_train, x_train_real, x_test, y_test, x_test_real, name)

    print(da.categorical_confusion_matrix(cvnn.predict(x_test), y_test))
    print(da.categorical_confusion_matrix(rvnn.predict(x_test_real), y_test))

    set_trace()
    """

__author__ = 'J. Agustin BARRACHINA'
__version__ = '0.0.14'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
