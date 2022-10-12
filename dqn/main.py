from tqdm import tqdm

if __name__ == "__main__":
    print("dqn")

    for i in tqdm(range(5_000_000)):
        pass


"""

Preprocessing
========================================
- Atari 2600 frames: 210x160 with 128-colour palette

1. to encode a single frame:
    - maximum value for each pixel colour value over the frame being encoded and the previous frame
2. extract the Y channel (aka luminance) from the RGB frame
3. rescale it to 84x84

The function phi applies this preprocessing to the m most recent frames and stacks them to produce the input to the Q-function
-> m=4
________________________________________




"""
