import torch
from .noise import NoiseGenerator
from .network import UNet
from snake.game import Direction


def prepare_input_train(
    noise_generator: NoiseGenerator, frames: list[torch.Tensor], action: int
):
    frames = [(frame * 2 - 1) for frame in frames]
    channel_shape = [1] + list(frames[0].shape[1:])

    curr = frames[-1]
    past = torch.cat(frames[:-1], dim=0)
    past += (
        noise_generator.normal.sample(past.shape).to(
            device=noise_generator.params.device
        )
        * 0.01
    )

    t = noise_generator.sample_t()
    noise_frame, noise = noise_generator.apply_noise(curr, t)

    t = torch.full(channel_shape, t / noise_generator.params.steps).to(
        device=noise_generator.params.device
    )
    action = torch.full(channel_shape, action / 4).to(
        device=noise_generator.params.device
    )

    prepared_input = torch.cat([past, noise_frame, t, action], dim=0)
    return prepared_input, noise


def prepare_input_eval(
    noise_generator: NoiseGenerator,
    history: list[torch.Tensor],
    action: int,
    t: int,
    frame: torch.Tensor | None = None,
):
    history = [(history_frame * 2 - 1) for history_frame in history]
    channel_shape = [1] + list(history[0].shape[1:])

    past = torch.cat(history, dim=0)

    t = torch.full(channel_shape, t / noise_generator.params.steps).to(
        device=noise_generator.params.device
    )
    action = torch.full(channel_shape, action / 4).to(
        device=noise_generator.params.device
    )

    prepared_input = torch.cat([past, frame, t, action], dim=0)
    return prepared_input


def generate_image(
    noise_generator: NoiseGenerator,
    diffuzer: UNet,
    history: list[torch.Tensor],
    action: Direction | int,
    output_steps: bool = False,
    n_steps: int = 100,
):
    if isinstance(action, Direction):
        action = int(action.to_int())
    inference_range = range(
        0, noise_generator.params.steps, noise_generator.params.steps // n_steps
    )[::-1]

    diffuzer_training = diffuzer.training
    diffuzer.eval()

    frame = noise_generator.normal.sample((3, 15, 15)).to(
        device=noise_generator.params.device
    )
    frames = [frame]
    for t in inference_range:
        with torch.no_grad():
            prepared_input = prepare_input_eval(
                noise_generator, history, action, t, frame
            )
            predicted_noise = diffuzer(prepared_input.unsqueeze(0)).squeeze(0)
            frame = noise_generator.denoise_step_ddim(frame, predicted_noise, t)

            if output_steps:
                frames.append(frame)
            else:
                frames[0] = frame

    frames = [
        frame + noise_generator.normal.sample(frame.shape) * 0.01 for frame in frames
    ]
    frames = [torch.clamp((frame + 1.0) / 2, 0.0, 1.0) for frame in frames]

    if diffuzer_training:
        diffuzer.train()

    if output_steps:
        return frames
    return frames[0]
