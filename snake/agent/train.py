import torch
from tqdm import tqdm
from dataclasses import dataclass
from snake.game.engine import Engine, Direction, StepState
from .network import SnakeNet
from pathlib import Path
from math import exp


@dataclass
class TrainingParameters:
    device: torch.device = torch.device("cpu")
    n_epochs: int = 100_000
    learning_rate: float = 0.005
    weight_decay: float = 0.0
    max_steps: int = 100
    save_interval: int = 250


def direction_from_index(index: torch.Tensor | int) -> Direction | None:
    match int(index):
        case 0:
            return Direction.UP
        case 1:
            return Direction.DOWN
        case 2:
            return Direction.LEFT
        case 3:
            return Direction.RIGHT
        case 4:
            return None


def discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        discounted_rewards.append(R)
    discounted_rewards = torch.tensor(discounted_rewards[::-1])
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        discounted_rewards.std() + 1e-5
    )
    return discounted_rewards


def train(params: TrainingParameters = TrainingParameters()):
    network = SnakeNet().to(device=params.device)
    optimizer = torch.optim.Adam(
        params=network.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )

    running_loss = 0.0
    running_apples = []
    best_avg_apples = 0.0
    for epoch in tqdm(range(params.n_epochs), "Training Snake RL Agent"):
        log_probs = []
        rewards = []

        engine = Engine()

        prev_distance = engine.get_distance_head_to_apple()
        no_apple_time = 0
        apples = 0
        steps_left = params.max_steps
        while steps_left > 0:
            probs = network(engine.get_description_vector().to(device=params.device))
            distribution = torch.distributions.Categorical(probs)
            sample = distribution.sample()
            direction = direction_from_index(sample)
            state = engine.step(direction)

            distance = engine.get_distance_head_to_apple()
            log_probs.append(distribution.log_prob(sample))
            match state:
                case StepState.APPLE:
                    apples += 1
                    rewards.append(exp(6.5 + apples * 0.7))
                    steps_left += 25
                    no_apple_time = 0
                case StepState.ALIVE:
                    reward = 1.0 if distance < prev_distance else 0.1
                    reward -= no_apple_time * 0.01
                    rewards.append(reward)
                    no_apple_time += 1
                    pass
                case StepState.DEAD:
                    rewards.append(-100.0)
                    break

            prev_distance = distance
            steps_left -= 1

        running_apples.append(apples)
        rewards = discounted_rewards(rewards).to(device=params.device)

        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(device=params.device)
        for log_prob, reward in zip(log_probs, rewards):
            loss += -log_prob * reward
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

        if (epoch + 1) % 250 == 0:
            avg_apples = torch.mean(torch.Tensor(running_apples))
            print(f"Loss ({running_loss:.4f}); Avg. apples ({avg_apples:.4f})")
            running_loss = 0.0
            running_apples = []

            best_path = Path(__file__).parent / "models/best.pt"
            deatiled_path = (
                Path(__file__).parent
                / f"models/detailed/avg{epoch}_{avg_apples:.4f}.pt"
            )

            torch.save(network.state_dict(), deatiled_path)
            if avg_apples > best_avg_apples:
                best_avg_apples = avg_apples
                torch.save(network.state_dict(), best_path)


if __name__ == "__main__":
    train()
