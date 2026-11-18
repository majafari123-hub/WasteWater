import numpy as np

from bsm1_lab.env import BSM1Env
from bsm1_lab.mock_backend import MockBackend
from bsm1_lab.observation import ObservationModel, ObservationSpec


env = BSM1Env(
    MockBackend(),
    ObservationModel(ObservationSpec(indices=(0, 2), noise_std=0.0)),
)
observation, mask, info = env.reset(seed=7)
assert observation.tolist() == [1.0, 3.0]
assert info["ground_truth"].tolist() == [1.0, 2.0, 3.0, 4.0]
step = env.step(np.array([0.5, -0.25]))
assert step.observation.tolist() == [1.5, 3.0]
print("OK: privileged state and partial observation are separated")
