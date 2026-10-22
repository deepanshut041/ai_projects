#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

std::pair<std::size_t, std::size_t> shape(const Matrix& matrix) {
    if (matrix.empty() || matrix.front().empty()) {
        throw std::invalid_argument("A matrix must be nonempty");
    }
    const auto width = matrix.front().size();
    for (const auto& row : matrix) {
        if (row.size() != width) {
            throw std::invalid_argument("Matrix rows must have equal lengths");
        }
    }
    return {matrix.size(), width};
}

Matrix transpose(const Matrix& input) {
    const auto [rows, columns] = shape(input);
    Matrix output(columns, Vector(rows));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            output[column][row] = input[row][column];
        }
    }
    return output;
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    const auto [rows, inner] = shape(left);
    const auto [right_rows, columns] = shape(right);
    if (inner != right_rows) {
        throw std::invalid_argument("Inner dimensions do not match");
    }
    Matrix result(rows, Vector(columns, 0.0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            for (std::size_t k = 0; k < inner; ++k) {
                result[row][column] += left[row][k] * right[k][column];
            }
        }
    }
    return result;
}

Matrix linear(const Matrix& inputs, const Matrix& weights, const Vector& bias) {
    auto result = matmul(inputs, weights);
    if (bias.size() != result.front().size()) {
        throw std::invalid_argument("Bias width differs from output width");
    }
    for (auto& row : result) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            row[column] += bias[column];
        }
    }
    return result;
}

Vector stable_softmax(const Vector& logits) {
    if (logits.empty()) {
        throw std::invalid_argument("Softmax requires nonempty logits");
    }
    for (const auto value : logits) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Softmax requires finite logits");
        }
    }
    const auto largest = *std::max_element(logits.begin(), logits.end());
    Vector result;
    double denominator = 0.0;
    for (const auto value : logits) {
        result.push_back(std::exp(value - largest));
        denominator += result.back();
    }
    for (auto& value : result) {
        value /= denominator;
    }
    return result;
}

struct CategoricalResult {
    double loss;
    Vector probabilities;
    Vector gradient;
};

CategoricalResult categorical_loss(const Vector& logits, std::size_t target) {
    auto probabilities = stable_softmax(logits);
    if (target >= logits.size()) {
        throw std::invalid_argument("Target is outside the vocabulary");
    }
    const auto largest = *std::max_element(logits.begin(), logits.end());
    double exponential_sum = 0.0;
    for (const auto value : logits) {
        exponential_sum += std::exp(value - largest);
    }
    auto gradient = probabilities;
    gradient[target] -= 1.0;
    return {largest + std::log(exponential_sum) - logits[target], probabilities, gradient};
}

struct AdamState {
    unsigned step = 0;
    double first_moment = 0.0;
    double second_moment = 0.0;
};

double adam_step(double parameter, double gradient, AdamState& state,
                 double learning_rate = 0.1, double beta1 = 0.5,
                 double beta2 = 0.5, double epsilon = 1e-6) {
    if (!(learning_rate > 0) || !(epsilon > 0) || !(beta1 >= 0 && beta1 < 1) || !(beta2 >= 0 && beta2 < 1)) {
        throw std::invalid_argument("Invalid Adam hyperparameters");
    }
    if (!std::isfinite(parameter) || !std::isfinite(gradient) || !std::isfinite(state.first_moment)
        || !std::isfinite(state.second_moment) || state.second_moment < 0) {
        throw std::invalid_argument("Adam requires finite values and nonnegative second moments");
    }
    ++state.step;
    state.first_moment = beta1 * state.first_moment + (1 - beta1) * gradient;
    state.second_moment = beta2 * state.second_moment + (1 - beta2) * gradient * gradient;
    const auto corrected_first = state.first_moment / (1 - std::pow(beta1, state.step));
    const auto corrected_second = state.second_moment / (1 - std::pow(beta2, state.step));
    return parameter - learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon);
}

void require_close(double actual, double expected, double tolerance = 1e-7) {
    if (std::abs(actual - expected) > tolerance || !std::isfinite(actual)) {
        throw std::runtime_error("A known-answer or finite-difference check failed");
    }
}

int main() {
    const Matrix inputs{{1, 2, 3}, {4, 5, 6}};
    const Matrix weights{{7, 8}, {9, 10}, {11, 12}};
    const Matrix expected{{58, 64}, {139, 154}};
    const auto result = matmul(inputs, weights);
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t column = 0; column < 2; ++column) {
            require_close(result[row][column], expected[row][column]);
        }
    }
    const auto probability = stable_softmax({1000, 1000 + std::log(3.0)});
    require_close(probability[0], 0.25);
    require_close(probability[1], 0.75);

    Vector logits{-0.7, 0.5, 1.2};
    const auto categorical = categorical_loss(logits, 1);
    constexpr double h = 1e-5;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        auto plus = logits;
        auto minus = logits;
        plus[index] += h;
        minus[index] -= h;
        const auto numerical = (categorical_loss(plus, 1).loss - categorical_loss(minus, 1).loss) / (2 * h);
        require_close(categorical.gradient[index], numerical);
    }
    require_close(categorical_loss({0, -1000}, 1).loss, 1000);

    const auto scalar_loss = [](double weight) {
        const double prediction = 2 * weight + (-1) * (-1) + 0;
        const double error = prediction - 1;
        return 0.5 * error * error;
    };
    require_close((scalar_loss(0.5 + h) - scalar_loss(0.5 - h)) / (2 * h), 2);

    AdamState state;
    const auto first_parameter = adam_step(1, 2, state);
    require_close(first_parameter, 1 - 0.1 * 2 / (2 + 1e-6));
    require_close(state.first_moment, 1);
    require_close(state.second_moment, 2);
    const auto second_parameter = adam_step(first_parameter, -1, state);
    require_close(second_parameter, first_parameter);
    require_close(state.first_moment, 0);
    require_close(state.second_moment, 1.5);
    std::cout << "{\"matrix_product\":[[" << result[0][0] << "," << result[0][1]
              << "],[" << result[1][0] << "," << result[1][1]
              << "]],\"probabilities\":[" << probability[0] << "," << probability[1]
              << "],\"central_differences\":true,\"adam_steps\":" << state.step << "}\n";
    return 0;
}
