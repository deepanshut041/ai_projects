// Explicit row-major storage, stable probabilities and one analytic update.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<double> softmax(const std::vector<double>& x) {
    if (x.empty()) throw std::invalid_argument("empty softmax");
    const auto peak = *std::max_element(x.begin(), x.end());
    std::vector<double> p;
    double sum = 0;
    for (const auto value : x) { p.push_back(std::exp(value - peak)); sum += p.back(); }
    for (auto& value : p) value /= sum;
    return p;
}

int main() {
    const std::vector<double> values{2, 4, 8};
    std::vector<double> causal;
    for (std::size_t i = 0; i < values.size(); ++i) {
        double sum = 0;
        for (std::size_t j = 0; j <= i; ++j) sum += values[j];
        causal.push_back(sum / static_cast<double>(i + 1));
    }
    // [B,T,D] flat offset: ((b * T) + t) * D + d.
    const std::size_t b = 1, t = 2, d = 3, T = 4, D = 8;
    const auto offset = (b * T + t) * D + d;
    const auto p = softmax({1000, 1000});
    const double w = 2, x = 3, target = 1;
    const auto grad = 2 * (w * x - target) * x;
    std::cout << std::fixed << std::setprecision(9)
              << "{\"offset\":" << offset
              << ",\"cross_entropy\":" << -std::log(p[0])
              << ",\"gradient\":" << grad
              << ",\"updated_weight\":" << w - 0.01 * grad
              << ",\"causal\":[" << causal[0] << ',' << causal[1] << ',' << causal[2]
              << "]}" << '\n';
}
