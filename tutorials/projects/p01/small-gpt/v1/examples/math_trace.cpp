// C++17: index one [B,H,T,d] element and compute a token-weighted loss.
#include <cassert>
#include <cmath>
#include <iostream>
int main() {
    const int H=4, T=3, d=2, b=1, h=2, t=1, c=0;
    const int offset = ((b*H+h)*T+t)*d+c;
    assert(offset == 38);
    const double loss = -(std::log(.5)+std::log(.25)+std::log(.125))/3.;
    assert(std::abs(std::exp(loss)-4.) < 1e-12);
    const double gradient_mean = (2.*1.+1.*4.)/3.;
    std::cout << "{\"offset\":" << offset << ",\"perplexity\":" << std::exp(loss)
              << ",\"weighted_mean\":" << gradient_mean << "}\n";
}
