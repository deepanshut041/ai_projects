"""One query, two keys: repeat the dimensionality experiment without packages."""
import math
import random
from row_math import dot, softmax


def inspect(width, rng, positive=False):
    draw = rng.random if positive else lambda: rng.gauss(0.0, 1.0)
    query = [draw() for _ in range(width)]
    keys = [[draw() for _ in range(width)] for _ in range(2)]
    scores = [dot(query, key) for key in keys]
    scaled = [score / math.sqrt(width) for score in scores]
    return scores, softmax(scores), scaled, softmax(scaled)


if __name__ == '__main__':
    for positive in [False, True]:
        print('Uniform [0,1]' if positive else 'Normal mean 0, variance 1')
        rng = random.Random(17)
        for width in [1, 4, 64, 1000, 4096]:
            raw, weights, scaled, scaled_weights = inspect(width, rng, positive)
            assert abs(sum(weights) - 1.0) < 1e-12
            assert abs(sum(scaled_weights) - 1.0) < 1e-12
            print(f'd={width:4} Q=(1,{width}) K=(2,{width}) '
                  f'scores={[round(x, 3) for x in raw]} '
                  f'weights={[round(x, 4) for x in weights]} '
                  f'scaled={[round(x, 3) for x in scaled]} '
                  f'new_weights={[round(x, 4) for x in scaled_weights]}')
