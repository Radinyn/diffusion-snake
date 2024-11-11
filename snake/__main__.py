import pygame
import torch
from .game.world import Tile
from .game.engine import Engine, Direction
from .diffusion.network import UNet
from PIL.Image import Image, Resampling
from pathlib import Path
import torchvision.transforms as transforms
from .diffusion.noise import NoiseGenerator, NoiseParameters
from .diffusion.utils import generate_image
from random import randint

tensor_to_image = transforms.ToPILImage()

HEIGHT = 500
WIDTH = 500
STEP_EACH_MS = 500
SUPPORTED = False

if SUPPORTED:
    N_STEPS = 30
    AVG_SIZE = 5
else:
    N_STEPS = 12
    AVG_SIZE = 1


def load_diffuzer() -> UNet:
    model = UNet()
    model_path = Path(__file__).parent / "diffusion/models/best.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def image_to_surface(image: Image) -> pygame.Surface:
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode).convert()


def get_direction() -> Direction | None:
    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        return Direction.LEFT
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        return Direction.RIGHT
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        return Direction.UP
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        return Direction.DOWN
    return None


def restore_apples(frame: torch.Tensor):
    if not SUPPORTED:
        return

    def count_apples():
        mask = torch.abs(frame - Tile.APPLE.all(frame.shape[1]))
        mask = torch.sum(mask, dim=0) < 1e-2
        return len(mask.nonzero())

    if count_apples() < 1:
        mask = torch.abs(frame - Tile.BACKGROUND.all(15))
        mask = torch.sum(mask, dim=0) < 1e-1
        indices = mask.nonzero()

        if len(indices) == 0:
            return

        index = randint(0, len(indices) - 1)
        i, j = indices[index]
        frame[:, i, j] = Tile.APPLE.value


if __name__ == "__main__":
    pygame.init()

    STEP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(STEP_EVENT, STEP_EACH_MS)
    direction = Direction.UP

    device = torch.device("cpu")
    diffuzer = load_diffuzer().to(device=device)
    noise = NoiseGenerator(params=NoiseParameters.default(device))

    history = []
    engine = Engine()
    for _ in range(3):
        frame = engine.get_frame().clone()
        frame += noise.normal.sample(frame.shape) * 0.01
        frame = torch.clamp(frame, 0.0, 1.0)
        history.append(frame)
        engine.step(direction)

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption("Snake")

    running = True
    started = False
    while running:
        direction = get_direction() or direction
        pygame.display.set_caption(f"Snake ({direction})")

        if not started:
            started = any(pygame.key.get_pressed())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == STEP_EVENT:
                if started:
                    avg = torch.empty(AVG_SIZE, *frame.shape)
                    for i in range(AVG_SIZE):
                        next_frame = generate_image(
                            noise_generator=noise,
                            diffuzer=diffuzer,
                            history=history,
                            action=direction,
                            n_steps=N_STEPS,
                        )

                        avg[i] = next_frame

                    next_frame = torch.median(avg, dim=0)[0]
                    restore_apples(next_frame)
                    history = history[1:] + [next_frame]

        image = tensor_to_image(history[-1]).resize((WIDTH, HEIGHT), Resampling.NEAREST)
        surface = image_to_surface(image)
        screen.blit(surface, (0, 0))
        pygame.display.flip()

    pygame.quit()
