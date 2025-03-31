# %%
import numpy as np


test_load = np.load(
    ".logs/78b53389-118b-43d7-888b-0a07b96eb2b3_Turmoil-v5_0_1743084930459.npz",
    allow_pickle=True,
)

game_data = test_load.f.arr_0

game_states = np.array([frame["obs_tp1"]["state"] for frame in game_data])


for i in range(len(game_states)):
    print(game_states[i,[57]])
# %%
