// Harness only. Upstream decoder is downloaded and prepared outside the source tree.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "decoder.inc"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    llama_backend_init(false);
    auto start = std::chrono::steady_clock::now();
    LanguageModelState state;
    if (!state.Initialize(argv[1])) return 3;
    auto elapsed = [](auto start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::cout << "load\t" << elapsed(start) << '\n';
    std::ifstream input(argv[2]);
    if (!input) return 4;
    std::string line;
    size_t index = 0;
    while (std::getline(input, line)) {
        auto split = line.find('\t');
        if (split == std::string::npos) return 5;
        std::string context = line.substr(0, split), prefix = line.substr(split + 1);
        start = std::chrono::steady_clock::now();
        std::vector<std::pair<float, std::string>> words;
        if (prefix.empty()) {
            words = state.PredictNextWord(context, {});
        } else {
            std::vector<TokenMix> mixes;
            // Use upstream's no-coordinate fallback. Non-ASCII prefixes are unsupported.
            bool supported = true;
            for (unsigned char c : prefix) {
                if (c < 'a' || c > 'z') { supported = false; break; }
                TokenMix mix{};
                mix.x = mix.y = -1.0f;
                for (auto &part : mix.mixes) part.token = state.specialTokens.LETTERS_TO_IDS[c - 'a'];
                mix.mixes[0].weight = 1.0f;
                mixes.push_back(mix);
            }
            if (supported) words = state.PredictCorrection(context, mixes, false, WordCapitalizeMode::IgnoredCapitals, {});
            // Preserve the JNI wrapper's exact-match preference before sorting.
            bool exact = false;
            for (const auto &word : words) exact |= isExactMatch(word.second, prefix);
            if (exact) {
                for (auto &word : words) {
                    if (!isExactMatch(word.second, prefix)) word.first -= 1.0f;
                }
            }
        }
        sortProbabilityPairVectorDescending(words);
        const auto ms = elapsed(start);
        std::cout << index++ << '\t' << std::setprecision(12) << ms;
        for (const auto &word : words) {
            if (word.second.find_first_of("\r\n\t") != std::string::npos) return 6;
            std::cout << '\t' << word.second;
        }
        std::cout << '\n';
    }
    return 0;
}
