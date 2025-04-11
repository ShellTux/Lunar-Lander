import argparse
from enum import StrEnum, auto

parser = argparse.ArgumentParser(description='Lunar Lander')
parser.add_argument(
    '--render',
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
    '--wind',
    type=bool,
    default=False,
    help="Enable the wind."
)
parser.add_argument(
    '--evolve',
    type=bool,
    default=True,
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
    render: RenderMode | None
    episodes: int
    gravity: float
    turbulence_power: float
    wind_power: float
    wind: bool
    evolve: bool
    generations: int
    crossover_probability: float
    mutation_probability: float

    def __init__(self, args: argparse.Namespace) -> None:
        assert args.render in RenderMode
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

        self.render = args.render if args.render != 'none' else None
        self.episodes = args.episodes
        self.gravity = args.gravity
        self.turbulence_power = args.turbulence_power
        self.wind_power = args.wind_power
        self.wind = args.wind
        self.evolve = args.evolve
        self.generations = args.generations
        self.crossover_probability = args.crossover_probability
        self.mutation_probability = args.mutation_probability

args = Args(parser.parse_args())
