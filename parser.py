# %%
import numpy as np


test_load = np.load(
    ".logs/576457e7-3a4e-4960-ba12-defe1bee9e68_Boxing-v5_0_1742909198954.npz",
    allow_pickle=True,
)

game_data = test_load.f.arr_0

game_states = np.array([frame["obs_tp1"]["state"] for frame in game_data])

print(game_data)
# %%
