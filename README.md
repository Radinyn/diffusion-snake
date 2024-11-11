# Diffusion Snake
<sub>Inspired by [DIAMOND](https://diamond-wm.github.io/)</sub>

This is a diffusion model simulating <b>Snake</b>.

The idea is to take some previous frames and the user input to predict the next frame.

It was never explicitly taught any Snake rules.

<sub>Note that apples are nondeterministic!</sub>

### Model in action:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/vQcCg20sRLs/0.jpg)](https://www.youtube.com/watch?v=vQcCg20sRLs)

### SUPPORTED version (more info below):

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/QOV0vhe40q4/0.jpg)](https://www.youtube.com/watch?v=QOV0vhe40q4)


## To play:

Download diffuzer weights (`best.pt`) from [here](https://drive.google.com/drive/folders/1V4p2XGTtjmiWkTjRYVZWCBkup8ImRYNu?usp=sharing).

Save them at: `snake/diffusion/models/best.pt`

To run do:
```bash
python -m snake
```

Tweak the `SUPPORTED` parameter in `__main__.py` to take the median of 5 model outputs and regenerate apples if they disappear. This makes the game smoother.

## To train:
```bash
# Train the RL Agent
python -m snake.agent.train # or use best_agent.pt

# Generate a dataset
python -m snake.dataset.dataset

# Train the diffuzer
python -m snake.diffusion.train
```

Training `best_agent.pt` agent took 14000 (short) epochs with learning rate 0.005.

Training `best.pt`diffuzer took 15 epochs with learning rate 0.001 and additional 5 epochs with learning rate 0.0001.
