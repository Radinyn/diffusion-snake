import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class NoiseParameters:
    steps: int
    beta: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    sqrt_alpha_bar: torch.Tensor
    sqrt_one_minus_alpha_bar: torch.Tensor

    noise_n: int = 15
    device: torch.device = torch.device("cpu")

    @classmethod
    def default(cls, device: torch.device = torch.device("cpu")) -> "NoiseParameters":
        steps = 100
        beta = torch.linspace(0.0001, 0.0075, steps).to(device=device)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)[:-1]
        alpha_bar = torch.cat((torch.tensor([1]).to(device=device), alpha_bar), axis=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        return cls(
            steps=steps,
            beta=beta,
            alpha=alpha,
            alpha_bar=alpha_bar,
            sqrt_alpha_bar=sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar=sqrt_one_minus_alpha_bar,
            device=device,
        )


class NoiseGenerator:
    def __init__(self, params: NoiseParameters):
        self.params = params
        self.normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        self.uniform = torch.distributions.Uniform(low=0, high=params.steps)
        self.shape = (3, params.noise_n, params.noise_n)

    def sample_noise(self, t: int | None = None) -> torch.Tensor:
        if t is None:
            t = self.params.steps - 1
        noise = self.normal.sample(self.shape).to(device=self.params.device)
        factor = self.params.sqrt_one_minus_alpha_bar[t]
        return noise * factor

    def apply_noise(
        self, frame: torch.Tensor, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = self.sample_noise(t)
        factor = self.params.sqrt_alpha_bar[t]
        noisy_frame = frame * factor + noise
        return noisy_frame, noise

    def sample_t(self) -> int:
        return int(self.uniform.sample())

    def denoise_step_ddim(
        self, frame: torch.Tensor, predicted_noise: torch.Tensor, t: int, sigma=0.001
    ) -> torch.Tensor:
        if t < 1:
            return frame

        denoised_frame = (
            frame - self.params.sqrt_one_minus_alpha_bar[t] * predicted_noise
        )
        denoised_frame /= self.params.sqrt_alpha_bar[t]
        denoised_frame *= torch.sqrt(self.params.alpha[t - 1])

        factor = torch.sqrt(1 - self.params.alpha[t - 1] - sigma * sigma)
        denoised_frame += factor * predicted_noise

        eps = self.normal.sample(self.shape).to(device=self.params.device)
        denoised_frame += sigma * eps
        return denoised_frame


if __name__ == "__main__":
    tensor_to_image = transforms.ToPILImage()

    noise = NoiseGenerator(params=NoiseParameters.default())

    image = torch.full((15, 15), fill_value=0.5)
    image[4:11, 4:11] = 1.0
    image = (image * 2) - 1
    image = torch.stack([image, image, image], dim=0)

    images = []

    for t in range(100):
        noisy_image, applied_noise = noise.apply_noise(image, t)
        noisy_image = (noisy_image + 1.0) / 2
        images.append(noisy_image)

    grid = make_grid(images, nrow=10)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(tensor_to_image(grid))
    plt.show()
