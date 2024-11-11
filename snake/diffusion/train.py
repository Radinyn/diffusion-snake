import torch
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from snake.dataset import SnakeDataset
from .noise import NoiseGenerator, NoiseParameters
from .network import UNet
from PIL.Image import Image
from .utils import prepare_input_train, generate_image

import torchvision.transforms as transforms
from torchvision.utils import make_grid

tensor_to_image = transforms.ToPILImage()


@dataclass
class TrainingParameters:
    dataset_path: str | Path = (
        Path(__file__).parent.parent / "dataset/data/dataset1000.pickle"
    )
    device: torch.device = torch.device("cpu")
    n_epochs: int = 500
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    batch_size: int = 32
    save_interval: int = 1000
    checkpoint_path: str | Path | None = (
        None  # Path(__file__).parent / "models/weights/unet154.pt"
    )
    output_suffix: str = ""  # "_tune"


def train(params: TrainingParameters = TrainingParameters()):
    noise_generator = NoiseGenerator(NoiseParameters.default(device=params.device))

    network = UNet().to(device=params.device)
    if params.checkpoint_path:
        network.load_state_dict(torch.load(params.checkpoint_path))

    dataset = SnakeDataset.load(params.dataset_path).to_device(params.device)

    optimizer = torch.optim.Adam(
        params=network.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    criterion = torch.nn.MSELoss().to(device=params.device)

    running_loss = 0
    batch_count = 0
    for epoch in range(params.n_epochs):
        inputs_batch = []
        noises_batch = []

        pbar = tqdm(
            iter(dataset), desc=f"Epoch ({epoch+1}/{params.n_epochs}) Loss: (Unknown)"
        )
        for action, frames in pbar:
            prepared_input, real_noise = prepare_input_train(
                noise_generator, frames, action
            )
            inputs_batch.append(prepared_input)
            noises_batch.append(real_noise)

            if len(inputs_batch) < params.batch_size:
                continue

            optimizer.zero_grad()

            inputs_batch = torch.stack(inputs_batch, dim=0)
            noises_batch = torch.stack(noises_batch, dim=0)

            predicted_batch = network(inputs_batch)
            loss = criterion(predicted_batch, noises_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            inputs_batch = []
            noises_batch = []

            batch_count += 1
            if batch_count % params.save_interval == 0:
                index = batch_count // params.save_interval
                pbar.set_description(
                    f"Epoch ({epoch+1}/{params.n_epochs}) Loss ({running_loss})"
                )
                running_loss = 0

                models_path = Path(__file__).parent / "models"
                images = frames + generate_image(
                    noise_generator, network, frames[:-1], action, output_steps=True
                )
                grid: Image = tensor_to_image(make_grid(images))
                grid.save(models_path / f"images{params.output_suffix}/ddim{index}.png")
                torch.save(
                    network.state_dict(),
                    models_path / f"weights{params.output_suffix}/unet{index}.pt",
                )


if __name__ == "__main__":
    train()
