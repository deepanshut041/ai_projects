"""Small arithmetic checks for P00 Lessons 3–11; Python standard library only.

Run after attempting the matching lesson's calculation. These checks verify
numbers and assumptions, not learner mastery or a trained model.
"""
import itertools
import json
import math
import unittest


def probabilities(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be positive; use argmax for greedy")
    shifted = [(z - max(logits)) / temperature for z in logits]
    terms = [math.exp(z) for z in shifted]
    return [term / sum(terms) for term in terms]


def mean(values):
    return sum(values) / len(values)


def variance(values):
    center = mean(values)
    return mean([(x - center) ** 2 for x in values])


def normalize(values, epsilon=1e-5):
    center, spread = mean(values), variance(values)
    return [(x - center) / math.sqrt(spread + epsilon) for x in values]


class LessonMath(unittest.TestCase):
    # Lessons 3–4: derivatives, probabilities and the update.
    def test_slope_approaches_derivative(self):
        loss = lambda w: (3 * w - 1) ** 2
        for h in (0.01, 0.001):
            self.assertAlmostEqual((loss(2 + h) - loss(2)) / h, 30 + 9 * h)
        h = 1e-5
        self.assertAlmostEqual((loss(2 + h) - loss(2 - h)) / (2 * h), 30)

    def test_partial_derivatives_and_update(self):
        loss = lambda w, b: (3 * w + b - 1) ** 2
        h = 1e-5
        dw = (loss(2 + h, 0) - loss(2 - h, 0)) / (2 * h)
        db = (loss(2, h) - loss(2, -h)) / (2 * h)
        self.assertAlmostEqual(dw, 30)
        self.assertAlmostEqual(db, 10)
        self.assertAlmostEqual(loss(2 - 0.01 * dw, -0.01 * db), 16)

    def test_log_probabilities_and_stability(self):
        for shift in (0, 1000, -1000):
            p = probabilities([shift, shift + math.log(3)])
            self.assertAlmostEqual(p[0], 0.25)
            self.assertAlmostEqual(p[1], 0.75)
        self.assertAlmostEqual(-math.log(p[1]), 0.2876820724517809)

    def test_cross_entropy_derivative(self):
        z, h = [0.2, -0.4, 0.8], 1e-5
        for target in range(3):
            for i in range(3):
                plus, minus = z.copy(), z.copy()
                plus[i] += h
                minus[i] -= h
                numerical = (-math.log(probabilities(plus)[target])
                             + math.log(probabilities(minus)[target])) / (2 * h)
                self.assertAlmostEqual(numerical, probabilities(z)[i] - (i == target))

    def test_adam_two_steps(self):
        m, v, theta = 0.0, 0.0, 2.0
        for t, g in enumerate([30.0, 0.0], start=1):
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mhat, vhat = m / (1 - 0.9 ** t), v / (1 - 0.999 ** t)
            theta -= 0.01 * mhat / (math.sqrt(vhat) + 1e-8)
            if t == 1:
                self.assertAlmostEqual(theta, 1.99)
        self.assertAlmostEqual(m, 2.7)
        self.assertAlmostEqual(v, 0.8991)
        self.assertLess(theta, 1.99)  # A zero new gradient does not erase momentum.

    def test_global_clipping(self):
        for g, expected in (([3.0, 4.0], [0.6, 0.8]), ([0.3, 0.4], [0.3, 0.4])):
            norm = math.sqrt(sum(x * x for x in g))
            clipped = [x * min(1, 1 / norm) for x in g]
            for actual, wanted in zip(clipped, expected):
                self.assertAlmostEqual(actual, wanted)

    # Lesson 6: data, conditional likelihood and reduction.
    def test_sequence_probability(self):
        ps = [0.5, 0.8, 0.9]
        self.assertAlmostEqual(math.prod(ps), 0.36)
        self.assertAlmostEqual(sum(-math.log(p) for p in ps), -math.log(0.36))
        self.assertAlmostEqual(mean([-math.log(p) for p in ps]), 0.3405504158439938)

    def test_valid_token_mean(self):
        first, second = [1, 1, 1], [3, 3]
        self.assertEqual(mean(first + second), 1.8)
        self.assertEqual(mean([mean(first), mean(second)]), 2.0)

    def test_data_space(self):
        self.assertEqual(sum(10 ** n for n in range(2, 7)), 1_111_100)

    # Lessons 7–8: positions, attention and normalization.
    def test_positions_and_power_identity(self):
        for position, expected in ((0, [0, 1, 0, 1]),
                                   (1, [0.841471, 0.540302, 0.010000, 0.999950])):
            actual = []
            for i in range(2):
                frequency = math.exp(-2 * i / 4 * math.log(10000))
                self.assertAlmostEqual(frequency, 10000 ** (-2 * i / 4))
                actual += [math.sin(position * frequency), math.cos(position * frequency)]
            for x, y in zip(actual, expected):
                self.assertAlmostEqual(x, y, places=6)

    def test_attention_mixture(self):
        weights = probabilities([1 / math.sqrt(2), 0])
        self.assertAlmostEqual(weights[0], 0.6697615493266569)
        self.assertAlmostEqual(weights[0] * 10 + weights[1] * 20, 13.30238450673343)
        self.assertEqual(sum(w * v for w, v in zip([0.25, 0.75], [2, 8])), 6.5)

    def test_variance_scaling_by_enumeration(self):
        # Enumerate independent fair +/-1 coordinates instead of random sampling.
        vectors = list(itertools.product([-1, 1], repeat=4))
        dots = [sum(a * b for a, b in zip(q, k)) for q in vectors for k in vectors]
        self.assertEqual(mean(dots), 0)
        self.assertEqual(variance(dots), 4)
        self.assertEqual(variance([x / 2 for x in dots]), 1)
        self.assertEqual(variance([x / 4 for x in dots]), 0.25)

    def test_layernorm_constant_and_distinct(self):
        self.assertEqual(mean([1, 3]), 2)
        self.assertEqual(variance([1, 3]), 1)
        self.assertEqual(normalize([5, 5]), [0, 0])
        result = normalize([1, 3])
        self.assertAlmostEqual(result[0], -1, places=5)
        self.assertAlmostEqual(result[1], 1, places=5)

    # Lessons 9–11: interpretation and the P01 bridge.
    def test_perplexity(self):
        loss = -math.log(0.36) / 3
        self.assertAlmostEqual(math.exp(loss), 1.4057211088362487)
        self.assertAlmostEqual(math.exp(-math.log(0.25)), 4)

    def test_warmup_update_numbers(self):
        lr = lambda update: 0.002 * min(update / 20, 1.0)
        self.assertEqual([lr(n) for n in [1, 2, 20, 21]], [0.0001, 0.0002, 0.002, 0.002])

    def test_dropout_expectation(self):
        p, x = 0.5, 4.0
        output_when_kept = x / (1 - p)
        self.assertEqual(output_when_kept, 8)
        self.assertEqual(p * 0 + (1 - p) * output_when_kept, 4)

    def test_pre_and_post_norm_zero_sublayer(self):
        x = [1, 3]
        pre = [a + 0 for a in x]
        post = normalize([a + 0 for a in x])
        self.assertEqual(pre, x)
        self.assertNotEqual(post, x)

    def test_gelu_from_gaussian_cdf(self):
        expected = [-0.15865525393145707, 0, 0.8413447460685429]
        for x, y in zip([-1, 0, 1], expected):
            cdf = (1 + math.erf(x / math.sqrt(2))) / 2
            self.assertAlmostEqual(x * cdf, y)

    def test_temperature_and_next_token_shift(self):
        p = probabilities([0, math.log(3)], temperature=0.5)
        self.assertAlmostEqual(p[0], 0.1)
        self.assertAlmostEqual(p[1], 0.9)
        with self.assertRaises(ValueError):
            probabilities([0, 1], temperature=0)
        tokens = [10, 20, 30, 40]
        self.assertEqual(list(zip(tokens[:-1], tokens[1:])), [(10, 20), (20, 30), (30, 40)])

    def test_parameters_and_decay(self):
        self.assertEqual(64 * 64, 4096)
        self.assertAlmostEqual(2 * (1 - 0.01 * 0.1), 1.998)
        V, D, F = 260, 32, 128
        count = 3 * V * D + V + 3 * (4 * D * D + 4 * D) + 2 * (2 * D * F + F + D) + 5 * 2 * D
        self.assertEqual(count, 54916)
        self.assertEqual(count * 4, 219664)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(LessonMath))
    print(json.dumps({"tests": result.testsRun, "passed": result.wasSuccessful()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
