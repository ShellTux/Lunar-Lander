from enum import StrEnum
import argparse
import gymnasium as gym

parser = argparse.ArgumentParser(description='Lunar Lander')
parser.add_argument(
    '--render-mode',
    choices=['none', 'human'],
    default='human',
    help="Set the render option to 'none' or 'human'."
)
parser.add_argument(
    '--episodes',
    type=int,
    default=1000,
    help="Set the number of episodes."
)
parser.add_argument(
    '--gravity',
    type=float,
    default=-10.0,
    help="Set the gravity."
)
parser.add_argument(
    '--turbulence-power',
    type=float,
    default=0.0,
    help="Set the turbulence power."
)
parser.add_argument(
    '--wind-power',
    type=float,
    default=15.0,
    help="Set the wind power."
)
parser.add_argument(
    '--enable-wind',
    action='store_true',
    help="Enable the wind."
)
parser.add_argument(
    '--evolve',
    action='store_true',
    help="Enable Evolution."
)
parser.add_argument(
    '--generations',
    type=int,
    default=100,
    help="Set the number of generations."
)
parser.add_argument(
    '--crossover-probability',
    type=float,
    default=.7,
    help="Set the number of crossover probability."
)
parser.add_argument(
    '--mutation-probability',
    type=float,
    default=.1,
    help="Set the number of generations."
)

class RenderMode(StrEnum):
    RenderNone = 'none'
    RenderHuman = 'human'

class Args:
    render_mode: RenderMode | None
    episodes: int
    gravity: float
    turbulence_power: float
    wind_power: float
    enable_wind: bool
    evolve: bool
    generations: int
    epochs: int
    crossover_probability: float
    mutation_probability: float

    def __init__(self, args: argparse.Namespace) -> None:
        assert args.render_mode in RenderMode
        assert type(args.episodes) == int
        assert type(args.gravity) == float
        assert type(args.turbulence_power) == float
        assert type(args.wind_power) == float
        assert type(args.evolve) == bool
        assert type(args.generations) == int
        assert type(args.crossover_probability) == float
        assert 0 <= args.crossover_probability <= 1
        assert type(args.mutation_probability) == float
        assert 0 <= args.mutation_probability <= 1

        self.render_mode = args.render_mode if args.render_mode != RenderMode.RenderNone else None
        self.episodes = args.episodes
        self.gravity = args.gravity
        self.turbulence_power = args.turbulence_power
        self.wind_power = args.wind_power
        self.enable_wind = args.enable_wind
        self.evolve = args.evolve
        self.epochs = self.generations = args.generations
        self.crossover_probability = args.crossover_probability
        self.mutation_probability = args.mutation_probability

    def make_env(
        self,
        *,
        render_mode: RenderMode | None = None,
        gravity: float | None = None,
        enable_wind: bool | None = None,
        wind_power: float | None = None,
        turbulence_power: float | None = None,
    ) -> gym.Env:
        return gym.make(
            'LunarLander-v3',
            render_mode=render_mode or self.render_mode,
            continuous=True,
            gravity=gravity or self.gravity,
            enable_wind=enable_wind or self.enable_wind,
            wind_power=wind_power or self.wind_power,
            turbulence_power=turbulence_power or self.turbulence_power,
        )

args = Args(parser.parse_args())
