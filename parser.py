# %%
import numpy as np
from pathlib import Path
# %%
logs_folder = ".logs"
logs_path = Path(logs_folder)
game_files=list(logs_path.glob(f"*Boxing-v5_[!0]*.npz"))

sessions_load = [
    np.load(
    file,
    allow_pickle=True,
    ) for file in game_files
]

session_data = [load.f.arr_0 for load in sessions_load]
# %%
for session in session_data:
    game_states = np.array([frame["obs_tp1"]["state"] for frame in session])
    game_actions = np.array([frame["action"] for frame in session])
    game_mode = session[0]["game_mode"]
    game_difficulty = session[0]["game_difficulty"]
    participant_id = session[0]["participant_id"]
    print(range(len(session)))

# for i in range(len(game_states)):
#     print(game_actions[i])
# print(np.max(game_states[:, 105:109], axis=0))
# %%
