from cvnn.montecarlo import run_gaussian_dataset_montecarlo

# run_gaussian_dataset_montecarlo(iterations=10, optimizer='sgd', shape_raw=[100, 40], dropout=None)
run_gaussian_dataset_montecarlo(iterations=10, optimizer='sgd', shape_raw=[100, 40], dropout=0.5)
# run_gaussian_dataset_montecarlo(iterations=10, optimizer='sgd', shape_raw=None, dropout=None)
# run_gaussian_dataset_montecarlo(iterations=10, optimizer='sgd', shape_raw=None, dropout=0.5)
