/// @file test_tokenizer.cpp
/// Our GGUF BPE tokenizer must produce exactly llama.cpp's token ids for this model (trusted reference),
/// and decode(encode(x)) must round-trip the bytes.
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "llama.h"
#include "runtime/tokenizer.h"

using namespace q38rt;

static std::string read_file(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) return "";
    std::string s;
    char b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [files...]\n", argv[0]); return 2; }
    GGUFParser g;
    if (!g.parse(argv[1])) return 1;
    Tokenizer tk;
    if (!tk.load(g)) return 1;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<std::pair<std::string, std::string>> cases = {
        {"english", "The capital of France is Paris. The capital of Germany is Berlin."},
        {"contractions", "I'm sure they'll say it's what we've done, isn't it? You'd think so. DON'T SHOUT."},
        {"numbers", "In 2026 there were 1234567 people and 3.14159 pies, 0x1F and 42nd place."},
        {"code", "def f(x):\n    return [i * 2 for i in range(x) if i % 3 == 0]\n\n\nclass A:\n\tpass\n"},
        {"whitespace", "a  b   c\n\n\n  d \t e\r\nf    "},
        {"french", "Le chat dort sur le canapé pendant que la pluie tombe doucement."},
        {"cjk", "猫在沙发上睡觉，外面下着小雨。日本語のテキストも試します。"},
        {"arabic", "القطة نائمة على الأريكة بينما يهطل المطر بهدوء."},
        {"hindi_marks", "नमस्ते दुनिया, यह एक परीक्षण है।"},
        {"thai_marks", "สวัสดีครับ ภาษาไทยมีวรรณยุกต์"},
        {"decomposed_accents", "e\xCC\x81te\xCC\x81 cafe\xCC\x81 nai\xCC\x88ve"},
        {"emoji", "Great job! 🎉🚀 👍🏽 family: 👨‍👩‍👧"},
        {"special_tokens", "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"},
        {"think_tags", "<think>\n\n</think>\n\nAnswer."},
        {"long_spaces", std::string(5000, ' ') + "x" + std::string(3000, '\n')},
    };
    for (int i = 2; i < argc; i++) cases.push_back({argv[i], read_file(argv[i])});
    // random byte soup of printable ASCII + some UTF-8 fragments
    {
        std::mt19937 rng(5);
        const char* frags[] = {"é", "ß", "猫", "🙂", " ", "\n", "\t", "'s", "123", "<|im_end|>", "abc", ".", "!!", "  "};
        for (int c = 0; c < 20; c++) {
            std::string s;
            for (int k = 0; k < 400; k++) {
                if (rng() % 3 == 0) s += frags[rng() % (sizeof frags / sizeof *frags)];
                else s += (char)(32 + rng() % 95);
            }
            cases.push_back({"random" + std::to_string(c), s});
        }
    }
    int fails = 0;
    for (auto& c : cases) {
        const std::string& s = c.second;
        std::vector<llama_token> ref(s.size() + 16);
        const int nr = llama_tokenize(vocab, s.data(), (int32_t)s.size(), ref.data(), (int32_t)ref.size(), false, true);
        ref.resize(nr > 0 ? nr : 0);
        const std::vector<int32_t> ours = tk.encode(s, true);
        size_t d = 0;
        while (d < ref.size() && d < ours.size() && ref[d] == ours[d]) d++;
        const bool same = ref.size() == ours.size() && d == ref.size();
        const bool rt = tk.decode(ours) == s;
        if (!same || !rt) {
            fails++;
            printf("%-20s llama %5zu ours %5zu %s roundtrip %s (first diff %zu: %d vs %d)\n", c.first.c_str(), ref.size(),
                   ours.size(), same ? "ids ok" : "IDS DIFFER", rt ? "ok" : "FAIL", d, d < ref.size() ? ref[d] : -1,
                   d < ours.size() ? ours[d] : -1);
        }
    }
    printf("test_tokenizer: %zu cases, %d failures\n", cases.size(), fails);
    llama_model_free(model);
    return fails ? 1 : 0;
}
