/// @file json.cpp
/// See json.h.

#include "server/json.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace q38rt {
namespace json {

const Value& Value::operator[](const std::string& key) const {
    static const Value null_value;
    if (type_ != Type::Object) return null_value;
    for (const auto& kv : o_) if (kv.first == key) return kv.second;
    return null_value;
}

bool Value::has(const std::string& key) const {
    if (type_ != Type::Object) return false;
    for (const auto& kv : o_) if (kv.first == key) return true;
    return false;
}

Value& Value::set(const std::string& key, Value v) {
    if (type_ != Type::Object) { type_ = Type::Object; o_.clear(); }
    for (auto& kv : o_) if (kv.first == key) return kv.second = std::move(v);
    o_.emplace_back(key, std::move(v));
    return o_.back().second;
}

namespace {
// Python repr of a float: shortest digits that round-trip; exponent form below 1e-4 or from 1e16 on
std::string py_float(double d) {
    if (std::isnan(d)) return "NaN";
    if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
    char buf[40];
    for (int p = 1; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p - 1, d);
        if (strtod(buf, nullptr) == d) break;
    }
    // buf = d.ddde[+-]XX with the shortest digits
    std::string m = buf;
    const size_t e = m.find('e');
    const int exp = atoi(m.c_str() + e + 1);
    std::string digits;
    for (size_t i = 0; i < e; i++) if (isdigit((unsigned char)m[i])) digits += m[i];
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const bool neg = d < 0;
    std::string out = neg ? "-" : "";
    if (exp < -4 || exp >= 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
        out += eb;
    } else if (exp < 0) {
        out += "0." + std::string(-exp - 1, '0') + digits;
    } else {
        const int intlen = exp + 1;
        if ((int)digits.size() <= intlen) out += digits + std::string(intlen - digits.size(), '0') + ".0";
        else out += digits.substr(0, intlen) + "." + digits.substr(intlen);
    }
    return out;
}
} // namespace

std::string quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;   // UTF-8 passes through
                }
        }
    }
    out += '"';
    return out;
}

void Value::dump_to(std::string& out, bool py) const {
    switch (type_) {
        case Type::Null: out += "null"; break;
        case Type::Bool: out += b_ ? "true" : "false"; break;
        case Type::Number: {
            char buf[40];
            if (int_ && std::fabs(n_) < 9.007199254740992e15) { snprintf(buf, sizeof buf, "%lld", (long long)n_); out += buf; break; }
            if (py) { out += py_float(n_); break; }
            if (!std::isfinite(n_)) { out += "null"; break; }
            if (n_ == std::floor(n_) && std::fabs(n_) < 1e15) snprintf(buf, sizeof buf, "%lld", (long long)n_);
            else snprintf(buf, sizeof buf, "%.17g", n_);
            out += buf;
            break;
        }
        case Type::String: out += quote(s_); break;
        case Type::Array: {
            out += '[';
            for (size_t i = 0; i < a_.size(); i++) {
                if (i) out += py ? ", " : ",";
                a_[i].dump_to(out, py);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            bool first = true;
            for (const auto& kv : o_) {
                if (!first) out += py ? ", " : ",";
                first = false;
                out += quote(kv.first);
                out += py ? ": " : ":";
                kv.second.dump_to(out, py);
            }
            out += '}';
            break;
        }
    }
}

std::string Value::dump() const {
    std::string out;
    dump_to(out, false);
    return out;
}

std::string Value::dump_py() const {
    std::string out;
    dump_to(out, true);
    return out;
}

struct Parser {
    const std::string& s;
    size_t i = 0;
    std::string err;

    bool fail(const char* msg) {
        if (err.empty()) err = std::string(msg) + " at offset " + std::to_string(i);
        return false;
    }
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }
    bool lit(const char* w) {
        size_t n = 0;
        while (w[n]) n++;
        if (s.compare(i, n, w) != 0) return false;
        i += n;
        return true;
    }
    static void utf8(uint32_t cp, std::string& out) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
    }
    bool hex4(uint32_t& v) {
        if (i + 4 > s.size()) return fail("truncated \\u escape");
        v = 0;
        for (int k = 0; k < 4; k++) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else return fail("bad hex digit");
        }
        return true;
    }
    bool string(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        i++;
        while (true) {
            if (i >= s.size()) return fail("unterminated string");
            const unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') { out += (char)c; continue; }
            if (i >= s.size()) return fail("truncated escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {   // high surrogate: expect the low half
                        uint32_t lo;
                        if (!lit("\\u") || !hex4(lo) || lo < 0xDC00 || lo >= 0xE000) return fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        return fail("lone low surrogate");
                    }
                    utf8(cp, out);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }
    bool number(double& out, bool& is_int) {
        const size_t start = i;
        is_int = true;
        if (i < s.size() && s[i] == '-') i++;
        if (i >= s.size()) return fail("bad number");
        if (s[i] == '0') i++;
        else if (s[i] >= '1' && s[i] <= '9') while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
        else return fail("bad number");
        if (i < s.size() && s[i] == '.') {
            is_int = false;
            i++;
            if (i >= s.size() || s[i] < '0' || s[i] > '9') return fail("bad fraction");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
        }
        if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
            is_int = false;
            i++;
            if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
            if (i >= s.size() || s[i] < '0' || s[i] > '9') return fail("bad exponent");
            while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
        }
        out = strtod(s.substr(start, i - start).c_str(), nullptr);
        return true;
    }
    bool value(Value& out, int depth) {
        if (depth > 64) return fail("nesting too deep");
        ws();
        if (i >= s.size()) return fail("unexpected end");
        const char c = s[i];
        if (c == '{') {
            i++;
            out = Value::object();
            ws();
            if (i < s.size() && s[i] == '}') { i++; return true; }
            while (true) {
                ws();
                std::string key;
                if (!string(key)) return false;
                ws();
                if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                i++;
                Value v;
                if (!value(v, depth + 1)) return false;
                out.set(key, std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            i++;
            out = Value::array();
            ws();
            if (i < s.size() && s[i] == ']') { i++; return true; }
            while (true) {
                Value v;
                if (!value(v, depth + 1)) return false;
                out.push(std::move(v));
                ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            std::string str;
            if (!string(str)) return false;
            out = Value(std::move(str));
            return true;
        }
        if (lit("true")) { out = Value(true); return true; }
        if (lit("false")) { out = Value(false); return true; }
        if (lit("null")) { out = Value(); return true; }
        double d;
        bool is_int;
        if (!number(d, is_int)) return false;
        out = Value(d);
        out.int_ = is_int;
        return true;
    }
};

bool Value::parse(const std::string& text, Value& out, std::string* error) {
    Parser p{text};
    Value v;
    bool ok = p.value(v, 0);
    if (ok) {
        p.ws();
        if (p.i != text.size()) ok = p.fail("trailing characters");
    }
    if (!ok) {
        if (error) *error = p.err;
        return false;
    }
    out = std::move(v);
    return true;
}

} // namespace json
} // namespace q38rt
