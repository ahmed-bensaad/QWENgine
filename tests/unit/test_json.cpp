/// @file test_json.cpp
/// JSON parser/writer: escapes (incl. surrogate pairs), numbers, nesting, round trips, rejection of bad input.

#include <cstdio>
#include <string>
#include "server/json.h"

using namespace q38rt;

static int g_fail = 0;
#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); g_fail++; } \
    } while (0)

int main() {
    json::Value v;
    std::string err;
    CHECK(json::Value::parse(R"({"a": [1, 2.5, -3e2, true, false, null], "s": "x\"y\\z\n\u00e9\ud83d\ude00", "o": {}})", v, &err));
    CHECK(v.is_object());
    CHECK(v["a"].items().size() == 6);
    CHECK(v["a"].items()[0].as_number() == 1);
    CHECK(v["a"].items()[1].as_number() == 2.5);
    CHECK(v["a"].items()[2].as_number() == -300);
    CHECK(v["a"].items()[3].as_bool() == true);
    CHECK(v["a"].items()[5].is_null());
    CHECK(v["s"].as_string() == "x\"y\\z\n\xC3\xA9\xF0\x9F\x98\x80");   // é and U+1F600 as UTF-8
    CHECK(v["o"].is_object() && v["o"].members().empty());
    CHECK(v["missing"].is_null());
    CHECK(v["missing"]["deeper"].is_null());

    // round trip
    json::Value w;
    CHECK(json::Value::parse(v.dump(), w, &err));
    CHECK(w.dump() == v.dump());

    // writer escapes control characters and quotes
    CHECK(json::quote(std::string("a\x01\"b\n", 5)) == "\"a\\u0001\\\"b\\n\"");
    json::Value n = json::Value::object();
    n.set("i", 42);
    n.set("f", 0.5);
    CHECK(n.dump() == "{\"i\":42,\"f\":0.5}");   // insertion order
    n.set("i", 7);                                 // replace keeps the position
    CHECK(n.dump() == "{\"i\":7,\"f\":0.5}");

    // Python json.dumps(ensure_ascii=False) style, as the chat template's tojson renders tool schemas
    json::Value py;
    CHECK(json::Value::parse(R"({"b": 1, "a": [1.0, 2.5, 1e-05, 1e16, 100000.0, -0.25, "\u00e9\n"], "c": {"x": null, "y": false}})", py, &err));
    CHECK(py.dump_py() == "{\"b\": 1, \"a\": [1.0, 2.5, 1e-05, 1e+16, 100000.0, -0.25, \"\xC3\xA9\\n\"], \"c\": {\"x\": null, \"y\": false}}");
    CHECK(py["b"].is_integer() && !py["a"].items()[0].is_integer());
    // more float reprs (expected strings from Python's json.dumps)
    CHECK(json::Value::parse("[0.1, 123456789.123, 1e-4, 9.5e15, 5e-324, 1.7976931348623157e308, -1e-7, 3]", py, &err));
    CHECK(py.dump_py() == "[0.1, 123456789.123, 0.0001, 9500000000000000.0, 5e-324, 1.7976931348623157e+308, -1e-07, 3]");

    // malformed inputs are rejected
    const char* bad[] = {"", "{", "[1,]", "{\"a\" 1}", "\"unterminated", "01", "1.", "tru", "{\"a\":1} x",
                         "\"\\ud800\"", "\"\\q\"", "\"a\nb\"", "[1 2]"};
    for (const char* b : bad) {
        json::Value t;
        const bool ok = json::Value::parse(b, t, &err);
        if (ok) printf("  accepted bad input: %s\n", b);
        CHECK(!ok);
    }
    // depth limit
    std::string deep(200, '[');
    deep += std::string(200, ']');
    CHECK(!json::Value::parse(deep, v, &err));

    if (g_fail) { printf("test_json: %d failures\n", g_fail); return 1; }
    printf("test_json: all checks passed\n");
    return 0;
}
