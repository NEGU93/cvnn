import tensorflow as tf
import data_processing as dp


def rvnn():
    m = 50
    n = 20
    x, y = dp.create_non_correlated_gaussian_noise(m, n)
    x, y = dp.randomize(x, y)
    x_real = dp.transform_to_real(x)
    x_train, y_train, x_test, y_test = dp.separate_into_train_and_test(x_real, y)



    import pdb; pdb.set_trace()
