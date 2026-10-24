#pragma once
/// @file json.h
/// Small, strict JSON value with a recursive-descent parser (RFC 8259: escapes incl. \uXXXX surrogate pairs ->
/// UTF-8, numbers, nesting depth limit) and two writers: compact (API responses) and Python-style (what the model's
/// chat template produces with `tojson`: ", " / ": " separators, non-ASCII kept, Python float repr). Objects keep
/// their members in insertion order (the chat template renders tool schemas in the client's key order).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace q38rt {
namespace json {

class Value {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : type_(Type::Bool), b_(b) {}
    Value(int v) : type_(Type::Number), n_(v), int_(true) {}
    Value(int64_t v) : type_(Type::Number), n_((double)v), int_(true) {}
    Value(double v) : type_(Type::Number), n_(v) {}
    Value(const char* s) : type_(Type::String), s_(s) {}
    Value(std::string s) : type_(Type::String), s_(std::move(s)) {}

    static Value array() { Value v; v.type_ = Type::Array; return v; }
    static Value object() { Value v; v.type_ = Type::Object; return v; }

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool(bool def = false) const { return type_ == Type::Bool ? b_ : def; }
    double as_number(double def = 0) const { return type_ == Type::Number ? n_ : def; }
    const std::string& as_string() const { static const std::string empty; return type_ == Type::String ? s_ : empty; }
    bool is_integer() const { return type_ == Type::Number && int_; }   // integer literal / constructed from an int
    const std::vector<Value>& items() const { return a_; }
    using Members = std::vector<std::pair<std::string, Value>>;
    const Members& members() const { return o_; }   // insertion order

    /// Object member (null value if missing or not an object).
    const Value& operator[](const std::string& key) const;
    bool has(const std::string& key) const;
    /// Set (replace in place, or append) an object member.
    Value& set(const std::string& key, Value v);
    Value& push(Value v) { type_ = Type::Array; a_.push_back(std::move(v)); return a_.back(); }

    /// Compact JSON.
    std::string dump() const;
    /// Python json.dumps(v, ensure_ascii=False) formatting.
    std::string dump_py() const;

    /// Parse a complete JSON text. Returns false (with a message) on any syntax error or trailing garbage.
    static bool parse(const std::string& text, Value& out, std::string* error = nullptr);

private:
    Type type_ = Type::Null;
    bool b_ = false;
    double n_ = 0;
    std::string s_;
    bool int_ = false;
    std::vector<Value> a_;
    Members o_;
    void dump_to(std::string& out, bool py) const;
    friend struct Parser;
};

/// JSON string literal (with quotes) for s.
std::string quote(const std::string& s);

} // namespace json
} // namespace q38rt
