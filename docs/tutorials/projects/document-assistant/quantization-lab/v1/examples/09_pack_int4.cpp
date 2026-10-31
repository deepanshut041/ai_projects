// Toy signed-nibble layout; this is not a GGUF Q4_K block.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

std::vector<std::uint8_t> pack(const std::vector<int>& codes) {
    std::vector<std::uint8_t> bytes((codes.size() + 1) / 2, 0);
    for (std::size_t i = 0; i < codes.size(); ++i) {
        if (codes[i] < -8 || codes[i] > 7) throw std::invalid_argument("INT4 range");
        const auto nibble = static_cast<std::uint8_t>(codes[i] & 15);
        bytes[i / 2] |= static_cast<std::uint8_t>(nibble << (4 * (i % 2)));
    }
    return bytes;
}

std::vector<int> unpack(const std::vector<std::uint8_t>& bytes, std::size_t count) {
    if (bytes.size() != (count + 1) / 2) throw std::invalid_argument("Wrong byte count");
    std::vector<int> codes;
    for (std::size_t i = 0; i < count; ++i) {
        const int nibble = (bytes[i / 2] >> (4 * (i % 2))) & 15;
        codes.push_back(nibble >= 8 ? nibble - 16 : nibble);
    }
    return codes;
}

int main() {
    const std::vector<int> codes{-8, -1, 0, 7, 3};
    const auto bytes = pack(codes);
    if (unpack(bytes, codes.size()) != codes) throw std::runtime_error("Round trip");
    bool rejected = false;
    try { (void)pack({8}); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("Out-of-range code accepted");
    std::cout << "{\"bytes\":[" << static_cast<int>(bytes[0]) << ','
              << static_cast<int>(bytes[1]) << ',' << static_cast<int>(bytes[2])
              << "],\"count\":5,\"round_trip\":true}\n";
}
