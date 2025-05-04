# %%
import numpy as np
from pathlib import Path
import pandas as pd
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
df = pd.DataFrame()
for session in session_data:
    game_states = np.array([frame["obs_tp1"]["state"] for frame in session])
    state_names = [f"ram_{i}" for i in range(game_states.shape[1])]
    game_actions = np.array([frame["action"] for frame in session])
    game_mode = session[0]["game_mode"]
    game_difficulty = session[0]["game_difficulty"]
    participant_id = session[0]["participant_id"]
    trial = session[0]["trial_number"]
    temp_df = pd.DataFrame(game_states, columns=state_names)
    temp_df["action"] = game_actions
    temp_df["mode"] = game_mode
    temp_df["difficulty"] = game_difficulty
    temp_df["participant_id"] = participant_id
    temp_df["trial"] = trial

    df = pd.concat([df, temp_df])
# %%

# %%
