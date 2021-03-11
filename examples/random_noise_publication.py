from cvnn.montecarlo import run_gaussian_dataset_montecarlo

run_gaussian_dataset_montecarlo(iterations=20, m=10000, n=128, param_list=None, validation_split=0.2,
                                epochs=150, batch_size=100, display_freq=1, optimizer='sgd',
                                shape_raw=[64], activation='cart_relu', debug=False, capacity_equivalent=False,
                                polar=False, do_all=True, dropout=0.5, tensorboard=False)
run_gaussian_dataset_montecarlo(iterations=20, m=10000, n=128, param_list=None, validation_split=0.2,
                                epochs=150, batch_size=100, display_freq=1, optimizer='sgd',
                                shape_raw=[128, 40], activation='cart_relu', debug=False, capacity_equivalent=False,
                                polar=False, do_all=True, dropout=0.5, tensorboard=False)
