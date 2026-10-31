#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

std::pair<std::size_t, std::size_t> shape(const Matrix& matrix) {
    if (matrix.empty() || matrix.front().empty()) {
        throw std::invalid_argument("Nonempty matrix required");
    }
    for (const auto& row : matrix) {
        if (row.size() != matrix.front().size()) {
            throw std::invalid_argument("Rectangular matrix required");
        }
    }
    return {matrix.size(), matrix.front().size()};
}

Matrix linear(const Matrix& rows, const Matrix& weights, const Vector& bias) {
    const auto [batch, inputs] = shape(rows);
    const auto [weight_rows, outputs] = shape(weights);
    if (weight_rows != inputs || bias.size() != outputs) {
        throw std::invalid_argument("Expected X[N,D], W[D,F], b[F]");
    }
    Matrix result(batch, Vector(outputs));
    for (std::size_t n = 0; n < batch; ++n) {
        for (std::size_t f = 0; f < outputs; ++f) {
            double total = bias[f];
            for (std::size_t d = 0; d < inputs; ++d) {
                total += rows[n][d] * weights[d][f];
            }
            result[n][f] = total;
        }
    }
    return result;
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    return linear(left, right, Vector(shape(right).second, 0));
}

Matrix transpose(const Matrix& values) {
    const auto [rows, columns] = shape(values);
    Matrix result(columns, Vector(rows));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < columns; ++c) {
            result[c][r] = values[r][c];
        }
    }
    return result;
}

double mean(const Vector& values) {
    if (values.empty()) {
        throw std::invalid_argument("Nonempty values required");
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double variance(const Vector& values) {
    const auto center = mean(values);
    double total = 0;
    for (const auto value : values) {
        total += (value - center) * (value - center);
    }
    return total / values.size();
}

double norm(const Vector& values) {
    return std::sqrt(std::inner_product(values.begin(), values.end(), values.begin(), 0.0));
}

Vector softmax(const Vector& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Nonempty logits required");
    }
    for (const auto value : logits) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Finite logits required");
        }
    }
    const auto largest = *std::max_element(logits.begin(), logits.end());
    Vector terms;
    for (const auto value : logits) {
        terms.push_back(std::exp(value - largest));
    }
    const auto sum = std::accumulate(terms.begin(), terms.end(), 0.0);
    for (auto& value : terms) {
        value /= sum;
    }
    return terms;
}

double nll(const Vector& logits, std::size_t target) {
    softmax(logits);
    if (target >= logits.size()) {
        throw std::invalid_argument("Target outside logits");
    }
    const auto largest = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (const auto value : logits) {
        sum += std::exp(value - largest);
    }
    return std::log(sum) - (logits[target] - largest);
}

struct AdamState {
    unsigned step = 0;
    double first = 0;
    double second = 0;
};

double adam(double parameter, double gradient, AdamState& state) {
    constexpr double beta1 = 0.9, beta2 = 0.999, rate = 0.01, epsilon = 1e-8;
    ++state.step;
    state.first = beta1 * state.first + (1 - beta1) * gradient;
    state.second = beta2 * state.second + (1 - beta2) * gradient * gradient;
    return parameter - rate * (state.first / (1 - std::pow(beta1, state.step))) /
        (std::sqrt(state.second / (1 - std::pow(beta2, state.step))) + epsilon);
}

void vector_json(const Vector& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void matrix_json(const Matrix& values) {
    std::cout << '[';
    for (std::size_t row = 0; row < values.size(); ++row) {
        if (row) std::cout << ',';
        vector_json(values[row]);
    }
    std::cout << ']';
}

int main() {
    std::cout << std::fixed << std::setprecision(9) << std::boolalpha;
    const auto output = linear({{1, 2}, {3, 4}}, {{1, 0, 2}, {0, 1, 3}}, {0, 0, 1});
    const auto rectangular = matmul({{1, 2, 3}, {4, 5, 6}}, {{7, 8}, {9, 10}, {11, 12}});
    const auto probabilities = softmax({0, std::log(3.0)});
    const Matrix inputs{{1, 2}, {3, 4}}, grad_output{{0.5, 1}, {1.5, 2}};
    const auto grad_weights = matmul(transpose(inputs), grad_output);
    const Vector grad_bias{grad_output[0][0] + grad_output[1][0], grad_output[0][1] + grad_output[1][1]};
    const auto loss = [](double weight) { return std::pow(3 * weight - 1, 2); };
    constexpr double h = 1e-5;
    const bool finite_match = std::abs((loss(2 + h) - loss(2 - h)) / (2 * h) - 30) < 1e-7;
    const double prediction = 2 * 3 + 0;
    const double error = prediction - 1;
    const Vector scalar_gradient{2 * error * 3, 2 * error};
    const Vector updated{2 - 0.01 * scalar_gradient[0], 0 - 0.01 * scalar_gradient[1]};
    const double updated_loss = std::pow(updated[0] * 3 + updated[1] - 1, 2);
    if (!finite_match || std::abs(nll({0, -1000}, 1) - 1000) > 1e-8
        || std::abs(nll({1e16, 1e16}, 1) - std::log(2.0)) > 1e-12
        || std::abs(nll({-1e16, -1e16}, 0) - std::log(2.0)) > 1e-12) {
        throw std::runtime_error("Derivative or stable-loss check failed");
    }
    AdamState state;
    const auto first = adam(2, 30, state);
    const auto second = adam(first, 0, state);
    std::cout << "{\"linear\":"; matrix_json(output);
    std::cout << ",\"rectangular\":"; matrix_json(rectangular);
    std::cout << ",\"mean\":" << mean({1, 3});
    std::cout << ",\"variance\":" << variance({1, 3});
    std::cout << ",\"norm\":" << norm({3, 4});
    std::cout << ",\"probabilities\":"; vector_json(probabilities);
    std::cout << ",\"nll\":" << nll({0, std::log(3.0)}, 1);
    std::cout << ",\"scalar_gradient\":"; vector_json(scalar_gradient);
    std::cout << ",\"sgd_parameters\":"; vector_json(updated);
    std::cout << ",\"sgd_loss\":" << updated_loss;
    std::cout << ",\"affine_gradient_weights\":"; matrix_json(grad_weights);
    std::cout << ",\"affine_gradient_bias\":"; vector_json(grad_bias);
    std::cout << ",\"adam_parameters\":"; vector_json({first, second});
    std::cout << ",\"adam_moments_after_two\":"; vector_json({state.first, state.second});
    std::cout << ",\"finite_difference_matches\":" << finite_match << "}\n";
}
