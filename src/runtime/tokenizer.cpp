/// @file tokenizer.cpp
/// Byte-level BPE with GGUF vocabulary/merges and the Frokenizer (re2c, Qwen3.5 regex) pre-tokenizer.

#include "runtime/tokenizer.h"
#include <algorithm>
#include <cstring>
#include <queue>
#include "common/utils.h"
#include "frokenizer_generated/lexer.hpp"

namespace q38rt {

namespace {

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the other 68 to U+0100.. in order.
struct ByteMap {
    uint32_t to_cp[256];
    std::unordered_map<uint32_t, uint8_t> to_byte;
    ByteMap() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            to_cp[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
            to_byte[to_cp[b]] = (uint8_t)b;
        }
    }
};
const ByteMap& bytemap() {
    static ByteMap m;
    return m;
}

std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F)); }
    return s;
}

inline uint64_t pair_key(int32_t a, int32_t b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

} // namespace

bool Tokenizer::load(const GGUFParser& g) {
    const GGUFValue* toks = g.find_kv("tokenizer.ggml.tokens");
    const GGUFValue* merges = g.find_kv("tokenizer.ggml.merges");
    const GGUFValue* types = g.find_kv("tokenizer.ggml.token_type");
    if (!toks || !merges || toks->arr_type != GGUFType::STRING || merges->arr_type != GGUFType::STRING) {
        LOG_ERROR("GGUF has no BPE vocabulary/merges");
        return false;
    }
    if (g.get_kv_string("tokenizer.ggml.model") != "gpt2" || g.get_kv_string("tokenizer.ggml.pre") != "qwen35") {
        LOG_ERROR("tokenizer is specialized for gpt2/qwen35");
        return false;
    }
    tokens_ = toks->arr_str;
    type_.assign(tokens_.size(), 1);
    if (types && types->arr_type == GGUFType::INT32 && types->arr_count == tokens_.size())
        memcpy(type_.data(), types->arr_raw.data(), tokens_.size() * 4);
    vocab_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); i++) vocab_.emplace(tokens_[i], (int32_t)i);
    const ByteMap& bm = bytemap();
    for (int b = 0; b < 256; b++) {
        auto it = vocab_.find(utf8(bm.to_cp[b]));
        if (it == vocab_.end()) { LOG_ERROR("byte %d has no token", b); return false; }
        byte_token_[b] = it->second;
    }
    merges_.reserve(merges->arr_str.size() * 2);
    for (size_t r = 0; r < merges->arr_str.size(); r++) {
        const std::string& m = merges->arr_str[r];
        const size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) continue;
        const std::string a = m.substr(0, sp), b = m.substr(sp + 1);
        auto ia = vocab_.find(a), ib = vocab_.find(b), ic = vocab_.find(a + b);
        if (ia == vocab_.end() || ib == vocab_.end() || ic == vocab_.end()) continue;
        merges_.emplace(pair_key(ia->second, ib->second), std::make_pair((int32_t)r, ic->second));
    }
    specials_.clear();
    for (size_t i = 0; i < tokens_.size(); i++)
        if (type_[i] == 3 || type_[i] == 4) specials_.push_back({tokens_[i], (int32_t)i});
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });
    for (auto& v : special_by_byte_) v.clear();
    for (size_t k = 0; k < specials_.size(); k++)
        if (!specials_[k].first.empty()) special_by_byte_[(uint8_t)specials_[k].first[0]].push_back((int)k);
    eos_ = (int32_t)g.get_kv_uint("tokenizer.ggml.eos_token_id", 248046);
    bos_ = (int32_t)g.get_kv_uint("tokenizer.ggml.bos_token_id", 248044);
    return true;
}

// Standard byte-level BPE on one pre-token: lowest merge rank first, leftmost on ties.
void Tokenizer::bpe_chunk(const char* s, size_t n, std::vector<int32_t>& out) const {
    if (n == 0) return;
    if (n == 1) { out.push_back(byte_token_[(uint8_t)s[0]]); return; }
    std::vector<int32_t> sym(n);
    std::vector<int> prev(n), next(n);
    for (size_t i = 0; i < n; i++) {
        sym[i] = byte_token_[(uint8_t)s[i]];
        prev[i] = (int)i - 1;
        next[i] = (int)i + 1;
    }
    struct Item { int32_t rank; int pos; int32_t l, r; };
    auto cmp = [](const Item& a, const Item& b) { return a.rank > b.rank || (a.rank == b.rank && a.pos > b.pos); };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);
    auto push = [&](int i) {
        const int j = next[i];
        if (i < 0 || j >= (int)n) return;
        auto it = merges_.find(pair_key(sym[i], sym[j]));
        if (it != merges_.end()) pq.push({it->second.first, i, sym[i], sym[j]});
    };
    for (int i = 0; i + 1 < (int)n; i++) push(i);
    std::vector<char> dead(n, 0);
    while (!pq.empty()) {
        const Item it = pq.top();
        pq.pop();
        const int i = it.pos;
        if (dead[i] || sym[i] != it.l) continue;
        const int j = next[i];
        if (j >= (int)n || dead[j] || sym[j] != it.r) continue;
        sym[i] = merges_.at(pair_key(it.l, it.r)).second;
        dead[j] = 1;
        next[i] = next[j];
        if (next[j] < (int)n) prev[next[j]] = i;
        if (prev[i] >= 0) push(prev[i]);
        push(i);
    }
    for (int i = 0; i < (int)n; i = next[i]) out.push_back(sym[i]);
}

void Tokenizer::encode_plain(const char* s, size_t n, std::vector<int32_t>& out) const {
    frokenizer::LexerState st;
    frokenizer::lexer_init(&st, s, n);
    frokenizer::Token tok;
    while (frokenizer::lexer_next(&st, &tok) != frokenizer::TOKEN_EOF) bpe_chunk(tok.start, tok.length, out);
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    out.reserve(text.size() / 3 + 8);
    if (!parse_special) {
        encode_plain(text.data(), text.size(), out);
        return out;
    }
    size_t seg = 0, i = 0;
    while (i < text.size()) {
        int32_t id = -1;
        size_t len = 0;
        for (int k : special_by_byte_[(uint8_t)text[i]]) {   // longest match first
            const auto& sp = specials_[k];
            if (sp.first.size() <= text.size() - i && memcmp(text.data() + i, sp.first.data(), sp.first.size()) == 0) {
                id = sp.second;
                len = sp.first.size();
                break;
            }
        }
        if (id < 0) { i++; continue; }
        encode_plain(text.data() + seg, i - seg, out);
        out.push_back(id);
        i += len;
        seg = i;
    }
    encode_plain(text.data() + seg, text.size() - seg, out);
    return out;
}

std::string Tokenizer::token_to_piece(int32_t id) const {
    if (id < 0 || id >= (int32_t)tokens_.size()) return "";
    const std::string& t = tokens_[id];
    if (is_special(id)) return t;
    const ByteMap& bm = bytemap();
    std::string out;
    for (size_t i = 0; i < t.size();) {
        uint32_t cp;
        const unsigned char c = (unsigned char)t[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        if (i + len > t.size()) break;
        cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)t[i + k] & 0x3F);
        auto it = bm.to_byte.find(cp);
        if (it != bm.to_byte.end()) out += (char)it->second;
        else out.append(t, i, len);
        i += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string s;
    for (int32_t id : ids) s += token_to_piece(id);
    return s;
}

} // namespace q38rt
