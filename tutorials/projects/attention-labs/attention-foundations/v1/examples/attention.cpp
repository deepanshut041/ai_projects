#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main() {
    using Row = std::array<double, 2>;
    const std::array<Row, 3> q{{{1, 0}, {0, 1}, {1, 1}}};
    const std::array<Row, 3> k{{{1, 1}, {0, 1}, {1, 2}}};
    const std::array<Row, 3> v{{{2, 0}, {0, -1}, {2, -1}}};
    std::array<Row, 3> output{};
    for (std::size_t i = 0; i < 3; ++i) {
        std::array<double, 3> scores{};
        for (std::size_t j = 0; j <= i; ++j)
            scores[j] = (q[i][0] * k[j][0] + q[i][1] * k[j][1]) / std::sqrt(2.0);
        const double maximum = *std::max_element(scores.begin(), scores.begin() + i + 1);
        std::array<double, 3> weights{};
        double total = 0;
        for (std::size_t j = 0; j <= i; ++j) {
            weights[j] = std::exp(scores[j] - maximum);
            total += weights[j];
        }
        for (std::size_t j = 0; j <= i; ++j)
            for (std::size_t d = 0; d < 2; ++d)
                output[i][d] += (weights[j] / total) * v[j][d];
    }
    std::cout << std::fixed << std::setprecision(6) << "{\"output\":[";
    for (std::size_t i = 0; i < 3; ++i) {
        if (i) std::cout << ',';
        std::cout << '[' << output[i][0] << ',' << output[i][1] << ']';
    }
    std::cout << "]}\n";
}
