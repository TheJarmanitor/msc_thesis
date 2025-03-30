# %%
import numpy as np
import cv2


"""
Boxing:
- track when difference between player score and enemy score is +- 5 (10s before and 5s after)
- record when player and enemy score stagnates for at least 5s
- track when player hits enemy correctly for the first time (5s before and 10s after)

Turmoil:
- track when player score stagnates for at least 5s
- track 5s before and 5s after player dies
- track 5s before and after PRIZE acquisition
- track first time TANK enemy is killed (the one that can be killed from behind only)

Word Zapper:
- track when word completion stagnates for at least 7s
- 5s before and 7 after completion symbols are hit
- 5s before and after player is hit by asteroid for the first time
- 5s before and 7 after player is hit by deadly asteroid or shuffling asteroid

In general:
- first 30s of tutorial for each game

"""


# %%
def create_video(frames, filename, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        filename,
        fourcc,
        fps,
        (width, height),
    )

    for j in frames:
        video.write(j)

    cv2.destroyAllWindows()
    video.release()


def remove_overlaps(segments):
    if not segments:
        return segments
    segments.sort(key=lambda x: x[0])
    result = [segments[0]]
    prev_start, prev_end = segments[0]
    for cur_start, cur_end in segments[1:]:
        # If current segment starts before or at the end of the previous segment,
        # adjust its start to prevent overlap.
        if cur_start <= prev_end:
            continue
            # cur_start = prev_end + 1
            # # Skip segment if it becomes too short or invalid.
            # if cur_start >= cur_end:
            #     continue
        result.append((cur_start, cur_end))
        prev_end = cur_end
    return result



####### boxing functions
# %%
def detect_box_score_difference_events(box_data, threshold=5, pre=10, post=5, fps=30):
    """Detect when the score difference reaches the threshold."""
    events = []
    for i in range(len(box_data)):
        player_score, enemy_score = box_data[i, 18], box_data[i, 19]
        if abs(player_score - enemy_score) >= threshold:
            start = max(0, i - (pre * fps))
            end = min(len(box_data), i + (post * fps))
            events.append((start, end))
    return remove_overlaps(events)
    # return events


# %%
def detect_box_score_stagnation_events(box_data, stagnation=5, fps=30):
    events = []
    start_idx = 0
    while start_idx < len(box_data):
        current_player = box_data[start_idx, 18]
        current_enemy = box_data[start_idx, 19]
        end_idx = start_idx
        # Extend while scores do not change
        while (
            end_idx < len(box_data)
            and box_data[end_idx, 18] == current_player
            and box_data[end_idx, 19] == current_enemy
        ):
            end_idx += 1
        if (end_idx - start_idx) >= stagnation * fps:
            events.append((start_idx, end_idx - 1))
        start_idx = end_idx
    return remove_overlaps(events)


def detect_box_first_hit_event(box_data, pre=5, post=10, fps=30):
    events = []
    for i in range(1, len(box_data)):
        if box_data[i - 1, 18] == 0 and box_data[i, 18] >= 1:
            start_frame = i - int(pre * fps)
            end_frame = i + int(pre * fps)
            events.append((start_frame, end_frame))
            break  # Only record the first hit
    return events


############ turmoil functions

def detect_


# %%

test_load = np.load(
    ".logs/576457e7-3a4e-4960-ba12-defe1bee9e68_Boxing-v5_0_1742909198954.npz",
    allow_pickle=True,
)

game_data = test_load.f.arr_0

game_states = np.array([frame["obs_tp1"]["state"] for frame in game_data], dtype="f")
game_frames = np.array([frame["obs_tp1"]["pixels"] for frame in game_data])
game_frames = game_frames[..., ::-1]


for i, (s_frame, e_frame) in enumerate(
    detect_first_hit_event(
        game_states,
    )
):
    event_frames = game_frames[[f for f in range(s_frame, e_frame)]]
    create_video(event_frames, f"score_difference_{i}.avi")
