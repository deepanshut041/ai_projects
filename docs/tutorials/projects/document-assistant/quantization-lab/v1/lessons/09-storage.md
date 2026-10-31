## 9. Store four-bit codes as four-bit codes

The quantizer gives us integers. A Python list of integers is useful for understanding them, but each element has object/container overhead. It does not establish compressed storage. We now write an explicit packed layout in C++, where the bytes are visible, then distinguish that layout from a real model container.

### Pack two nibbles into one byte

A nibble contains four bits. Our teaching format uses signed values -8 through 7, represented by the low four bits of two's-complement integers. This differs from Lesson 8's symmetric -7…7 quantizer: the packing container can hold -8 even when that particular quantizer never emits it. The first code occupies bits 0–3; the next occupies bits 4–7.

For codes `[-8,-1,0,7,3]`, the nibbles are hexadecimal `8,F,0,7,3`. The first byte is `0xF8=248`; the second is `0x70=112`; the last is `0x03=3`. The upper nibble of the last byte is padding. Without the original element count, unpacking would invent a sixth zero. A valid format therefore preserves count/shape, not just bytes.

To recover a nibble $u\in[0,15]$, return $u$ if $u<8$ and $u-16$ otherwise. This restores the signed range. Masking and shifting are layout operations; they do not apply the quantization scale. Reconstructing a weight still requires $\hat w=s(q-z)$.

### Complete C++ reference

Create `09_pack_int4.cpp`. `pack()` validates the signed range before masking; otherwise code 8 could silently become -8 on reload. It uses `std::uint8_t` for byte storage and widens to `int` for printing. `unpack()` checks the declared count against the byte count, then extracts each nibble in the same order. The final test includes an odd number of codes and a rejected out-of-range code.

[Download the complete reference](examples/09_pack_int4.cpp)

```cpp
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
```

From the scratch directory containing that file, run:

```bash
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic 09_pack_int4.cpp -o pack_int4
./pack_int4
```

Expected checked output (JSON formatting may differ):

```json
{
  "bytes": [
    248,
    112,
    3
  ],
  "count": 5,
  "round_trip": true
}
```

A wrong first byte usually means high/low nibble order changed. A wrong negative value points to sign reconstruction. An extra output value indicates lost shape/count metadata. These bugs can leave a model file structurally readable while corrupting its weights, so a load-success check alone is insufficient.

### Count scales and zero points as part of storage

Suppose groups hold $g$ values, payload uses $b$ bits per value, and each group stores $m$ metadata bytes. Ignoring padding for a moment, effective bits per weight are

$$b_{\mathrm{effective}}=b+\frac{8m}{g}.$$

At four payload bits, a two-byte scale and group size 32 produce $4+16/32=4.5$ bits per value. If the same block also stores a two-byte zero point, the total is $4+32/32=5$. For group size 128 with only a two-byte scale, it becomes 4.125. Larger groups reduce metadata cost while forcing more values to share a range; Lesson 15 compares that tradeoff under a cap.

A group containing 33 elements with nominal size 32 needs two groups. If the format pads a complete second block, payload can exceed `ceil(33*4/8)`. Tensor alignment, headers and unchanged higher-precision tensors add more. For actual artifact accounting, sum each stored tensor/block according to its format and compare the result with the file size, explaining any remaining container overhead.

### A real container is a different contract

The [GGUF specification](source:gguf) describes metadata and tensors in a model container, including offsets/alignment and quantization type identities. Our nibble code is **not** a GGUF writer or a Q4_K block encoder. Q4_K_M is a runtime preset whose actual tensor mix must be inspected. Do not rename this toy byte stream `.gguf` and expect a runtime to load it.

Disk compression also differs from inference arithmetic. A backend may unpack a small block on the fly, use specialized low-bit kernels, or expand a large amount into floating storage. Each choice changes workspace, bandwidth and throughput. A small file plus a large float expansion can satisfy a download constraint while failing the device memory goal.

[[practice:storage-transfer]]

We can now explain both the quantized values and their bytes. The next question is more important to the assistant: what does the weight reconstruction error do to the activations that select an answer token?
