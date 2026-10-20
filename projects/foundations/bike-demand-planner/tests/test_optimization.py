"""Known optimizer arithmetic, batch coverage, schedules and continuation state."""

import json
import math
import unittest

import numpy as np

from bike_demand_planner.optimization import BatchStream, TrialConfig, fit_trial, scheduled_rate
from bike_demand_planner.optimizers import OptimizerConfig, OptimizerState, initial_state, optimizer_step
from bike_demand_planner.regression import loss_and_gradients


X = np.array([[1., 0], [0, 1], [1, 1], [2, -1]])
Y = np.array([5., 2, 4, 8])


class OptimizationTests(unittest.TestCase):
    def test_sgd_and_classical_momentum_match_two_known_matrix_updates(self):
        for kind, expected in (("sgd", [1.8875, .81125, .6175]), ("momentum", [2.3375, .72125, .9325])):
            theta = np.array([1., 1, 0]); state = initial_state(3)
            for step in (1, 2):
                _, dw, db = loss_and_gradients(X, Y, theta[:2], theta[2])
                previous = theta.copy()
                theta, state = optimizer_step(theta, np.append(dw, db), state, OptimizerConfig(kind), .1)
                self.assertEqual(state.step, step)
                if step == 1:
                    np.testing.assert_allclose(theta, [1.5, .9, .35], atol=1e-12)
                    np.testing.assert_array_equal(previous, [1, 1, 0])
            np.testing.assert_allclose(theta, expected, atol=1e-12)

    def test_adam_constant_gradient_corrects_both_moments_at_steps_one_and_two(self):
        theta = np.array([1., -2]); gradient = np.array([2., -4]); state = initial_state(2)
        for step in (1, 2):
            theta, state = optimizer_step(theta, gradient, state, OptimizerConfig("adam"), .1)
            np.testing.assert_allclose(state.first / (1 - .9**step), gradient, atol=1e-12)
            np.testing.assert_allclose(state.second / (1 - .999**step), gradient**2, atol=1e-12)
            expected = [1 - step * .1 * 2 / (2 + 1e-8), -2 + step * .1 * 4 / (4 + 1e-8)]
            np.testing.assert_allclose(theta, expected, atol=1e-12)

    def test_adam_epsilon_is_outside_sqrt_and_state_buffers_do_not_alias_input(self):
        state = initial_state(1)
        theta, updated = optimizer_step([0], [1e-10], state, OptimizerConfig("adam"), .1)
        self.assertAlmostEqual(theta[0], -.1 * 1e-10 / (1e-10 + 1e-8), places=14)
        np.testing.assert_array_equal(state.first, [0])
        self.assertEqual(state.step, 0)
        restored = OptimizerState.from_dict(json.loads(json.dumps(updated.to_dict())), 1)
        np.testing.assert_array_equal(restored.first, updated.first)

    def test_adam_changing_gradient_keeps_history_instead_of_following_only_the_latest_sign(self):
        theta, state = optimizer_step([0], [2], initial_state(1), OptimizerConfig("adam"), .1)
        theta, state = optimizer_step(theta, [-1], state, OptimizerConfig("adam"), .1)
        self.assertAlmostEqual(state.first[0], .08, places=14)
        self.assertAlmostEqual(state.second[0], .004996, places=14)
        expected = -.1 * 2 / (2 + 1e-8) - .1 * (.08 / .19) / (math.sqrt(.004996 / .001999) + 1e-8)
        self.assertAlmostEqual(theta[0], expected, places=13)
        self.assertLess(theta[0], -.1)

    def test_epoch_coverage_retains_the_tail_and_shuffle_is_reproducible(self):
        a, b, changed = BatchStream(65, 32, 17), BatchStream(65, 32, 17), BatchStream(65, 32, 29)
        first_epoch = [a.next_indices() for _ in range(3)]
        self.assertEqual(list(map(len, first_epoch)), [32, 32, 1])
        np.testing.assert_array_equal(np.sort(np.concatenate(first_epoch)), np.arange(65))
        for expected in first_epoch:
            np.testing.assert_array_equal(b.next_indices(), expected)
        self.assertFalse(np.array_equal(changed.next_indices(), first_epoch[0]))
        second_epoch = np.concatenate([a.next_indices() for _ in range(3)])
        np.testing.assert_array_equal(np.sort(second_epoch), np.arange(65))
        self.assertFalse(np.array_equal(second_epoch, np.concatenate(first_epoch)))
        self.assertEqual(a.examples_seen, 130)

    def test_tail_gradient_uses_its_actual_count(self):
        y = np.array([1., 2, 3, 4, 10])
        config = TrialConfig(learning_rate=.1, batch_size=2, max_examples=5)
        result = fit_trial(np.zeros((5, 1)), y, config)
        expected = math.fsum(y) / 5
        order = result.state["batch_stream"]["order"]
        for start in (0, 2, 4):
            indices = order[start:start + 2]
            expected -= .1 * math.fsum(expected - y[i] for i in indices) / len(indices)
        self.assertAlmostEqual(result.parameters[-1], expected, places=12)
        self.assertEqual(result.last_batch_size, 1)
        self.assertEqual(result.state["optimizer"]["step"], 3)

    def test_inverse_time_schedule_tracks_examples_before_the_update(self):
        config = TrialConfig(learning_rate=.1, decay_examples=100)
        self.assertEqual(scheduled_rate(config, 0), .1)
        self.assertEqual(scheduled_rate(config, 100), .05)
        self.assertAlmostEqual(scheduled_rate(config, 200), .1 / 3)
        self.assertEqual(scheduled_rate(TrialConfig(learning_rate=.1), 1000), .1)

    def test_json_continuation_replays_parameters_moments_order_and_rng_exactly(self):
        for kind in ("sgd", "momentum", "adam"):
            config = TrialConfig(OptimizerConfig(kind), .01, 3, 17, max_examples=12)
            full = fit_trial(X, Y, config)
            paused = fit_trial(X, Y, config, pause_after_updates=1)
            self.assertEqual(paused.stop_reason, "paused")
            restored = fit_trial(X, Y, config, continuation=json.loads(json.dumps(paused.state)))
            np.testing.assert_array_equal(restored.parameters, full.parameters)
            self.assertEqual(restored.state["optimizer"], full.state["optimizer"])
            self.assertEqual(restored.state["batch_stream"], full.state["batch_stream"])
            self.assertEqual(restored.state["schedule"], full.state["schedule"])

    def test_update_time_example_and_divergence_stops_are_explicit(self):
        for config, reason in (
            (TrialConfig(max_examples=5, batch_size=3), "examples_budget"),
            (TrialConfig(max_updates=1), "updates_budget"),
            (TrialConfig(max_seconds=1e-15), "time_budget"),
            (TrialConfig(learning_rate=100, batch_size=4), "diverged"),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(fit_trial(X, Y, config).stop_reason, reason)

    def test_invalid_settings_dimensions_and_saved_orders_are_rejected(self):
        for settings in ({"learning_rate": float("nan")}, {"batch_size": 0}, {"max_updates": True},
                         {"max_seconds": 0}, {"seed": -1}, {"decay_examples": 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TrialConfig(**settings)
        with self.assertRaises(ValueError):
            optimizer_step([1, 2], [[1, 2]], initial_state(2), OptimizerConfig(), .1)
        with self.assertRaises(ValueError):
            optimizer_step([1, 2], [1, np.inf], initial_state(2), OptimizerConfig(), .1)
        invalid = BatchStream(5, 2, 17).to_dict()
        invalid["order"] = [0] * 5
        with self.assertRaises(ValueError):
            BatchStream.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
