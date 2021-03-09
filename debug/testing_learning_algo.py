import numpy as np
from pathlib import Path
from pdb import set_trace

path = Path("/home/barrachina/Documents/cvnn/log/montecarlo/2021/03March/09Tuesday/run-18h50m44")
init_weight = np.load(path / "initial_weights.npy", allow_pickle=True)
init_debug_weight = np.load(path / "initial_debug_weights.npy", allow_pickle=True)

complex_dict = {
    "init_weights": np.array(init_weight[0]),
    "gradients": np.load(path / "run/iteration0_model0_complex_network/gradients.npy", allow_pickle=True),
    "final_weights": np.load(path / "run/iteration0_model0_complex_network/final_weights.npy", allow_pickle=True)
}
real_dict = {
    "init_weights": init_weight[1]
}

assert len(complex_dict["init_weights"]) == len(complex_dict["gradients"]) == len(complex_dict["final_weights"])

for i_w, f_w, gr in zip(complex_dict["init_weights"], complex_dict["final_weights"], complex_dict["gradients"]):
    gr = gr.numpy()
    print(np.allclose(f_w, i_w - 0.01 * gr))
    set_trace()


