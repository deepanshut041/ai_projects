// Illustrative zero-logit attention, not a learned model.
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>

std::array<double, 3> attend(const std::array<double,3>& values, bool causal) {
    std::array<double,3> output{};
    for (std::size_t i=0; i<values.size(); ++i) {
        const auto count=causal ? i+1 : values.size();
        output[i]=std::accumulate(values.begin(), values.begin()+count, 0.0) / static_cast<double>(count);
    }
    return output;
}
int main() {
    const auto causal=attend({2,4,8},true);
    const auto unmasked=attend({2,4,8},false);
    const auto changed=attend({2,4,98},true);
    const std::array<int,4> tokens{10,20,30,40};
    std::cout << std::fixed << std::setprecision(9);
    auto print=[](const auto& values, std::size_t n) {
        std::cout << '[';
        for(std::size_t i=0;i<n;++i) {if(i)std::cout << ',';std::cout << values[i];}
        std::cout << ']';
    };
    std::cout << "{\"causal\":";print(causal,3);
    std::cout << ",\"unmasked\":";print(unmasked,3);
    std::cout << ",\"changed_prefix\":";print(changed,2);
    std::cout << ",\"inputs\":";print(tokens,3);
    std::cout << ",\"targets\":";print(std::array{tokens[1],tokens[2],tokens[3]},3);
    std::cout << "}\n";
}
