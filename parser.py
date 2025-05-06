# %%
import numpy as np
from pathlib import Path
import pandas as pd
import pyxdf

# %%
logs_folder = ".logs"
logs_path = Path(logs_folder)
game_files = list(logs_path.glob(f"P*-v5_[!0]*.npz"))


# %%
atari_dataframes = []
for file in game_files:
    with np.load(file, allow_pickle=True) as data:
        session = data.f.arr_0
        print(session)
        game_states = np.array([frame["obs_tp1"]["state"] for frame in session])
        state_names = [f"ram_{i}" for i in range(game_states.shape[1])]
        game_actions = np.array([frame["action"] for frame in session])
        game_name = session[0]["game_name"]
        game_mode = session[0]["game_mode"]
        game_difficulty = session[0]["game_difficulty"]
        participant_id = session[0]["participant_id"]
        trial = session[0]["trial_number"]
        temp_df = pd.DataFrame(game_states, columns=state_names)
        temp_df["participant_id"] = participant_id
        temp_df["game_name"] = game_name
        temp_df["trial"] = trial
        temp_df["mode"] = game_mode
        temp_df["difficulty"] = game_difficulty
        temp_df["action"] = game_actions
        temp_df = temp_df.rename_axis('frame').reset_index()
        temp_df = temp_df.loc[:, ["participant_id", "game_name", "trial", "mode", "difficulty", "frame", "action"] + state_names]
        atari_dataframes.append(temp_df)
        break

# %%
atari_df = pd.concat(atari_dataframes, ignore_index=True)
atari_df.to_csv(r"data/atari_logs.csv")

# %%

files_path = Path(".lsl")
xdf_files=list(files_path.glob(f"sub-*task-game_*_eeg.xdf"))

pxi_info = [
 "participant_id", 
 "game_name",  
 "trial", 
 "mode", 
 "difficulty", 
 "pxi_AA", 
 "pxi_CH", 
 "pxi_EC", 
 "pxi_GR", 
 "pxi_PF", 
 "pxi_AUT", 
 "pxi_CUR", 
 "pxi_IMM", 
 "pxi_MAS", 
 "pxi_MEA",  
 "pxi_ENJ",  
 "STIMES", 
 "TIMES"
]
pxi_dataframes = []

for file in xdf_files:
    streams, header = pyxdf.load_xdf(file, synchronize_clocks=True, verbose=True)

    # Identify EEG, GSR, and Eye Tracker streams
    # eeg_stream = next(s for s in streams if "eeg" in s["info"]["type"][0].lower())
    # eye_stream = next(s for s in streams if "pupil" or "gaze" in s["info"]["type"][0].lower())
    # atari_stream = next(s for s in streams if "game" in s["info"]["type"][0].lower())
    try:
        pxi_stream = next(s for s in streams if "survey" in s["info"]["type"][0].lower())
        pxi_data = np.array(pxi_stream["time_series"])
        temp_df = pd.DataFrame(pxi_data, columns=pxi_info)
        pxi_dataframes.append(temp_df)
    except:
        continue
# %%
pxi_df = pd.concat(pxi_dataframes, ignore_index=True)
pxi_df.to_csv(r"data/pxi_results.csv", index=False)


# %%
