from cvnn.montecarlo import run_gaussian_dataset_montecarlo

run_gaussian_dataset_montecarlo(iterations=500, m=10000, n=128, param_list=None,
                                epochs=150, batch_size=100, optimizer='sgd',
                                shape_raw=None, activation='cart_relu', debug=False,
                                polar=False, do_all=True, dropout=None)
