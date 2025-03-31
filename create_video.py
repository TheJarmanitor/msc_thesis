import cv2
import numpy as np


test_load = np.load(
    ".logs/78b53389-118b-43d7-888b-0a07b96eb2b3_Turmoil-v5_0_1743084930459.npz",
    allow_pickle=True,
)

game_data = test_load.f.arr_0


game_frames = np.array([frame["obs_tp1"]["pixels"] for frame in game_data])
game_frames = game_frames[..., ::-1]
print(game_frames.shape)

height, width, layers = game_frames[1].shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(
    "video.avi",
    fourcc,
    30,
    (width, height),
)


for j in game_frames:
    video.write(j)

cv2.destroyAllWindows()
video.release()
