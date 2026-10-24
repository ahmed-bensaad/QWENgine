#pragma once
/// @file http.h
/// Minimal HTTP/1.1 server: one thread per connection (the handler serializes access to the engine), Content-Length
/// bodies, plain responses or Server-Sent Events streams, CORS headers, Connection: close.

#include <functional>
#include <map>
#include <string>

namespace q38rt {
namespace http {

struct Request {
    std::string method, path, body;
    std::map<std::string, std::string> headers;   // lower-case names
};

class Responder {
public:
    explicit Responder(int fd) : fd_(fd) {}
    /// Complete response (once).
    bool send(int status, const std::string& content_type, const std::string& body);
    /// Start a text/event-stream response; then event() per message ("data: <payload>\n\n").
    bool start_events();
    bool event(const std::string& payload);
    /// Named event ("event: <name>\ndata: <payload>\n\n", as the Responses API streams).
    bool event(const std::string& name, const std::string& payload);
    bool sent() const { return sent_; }
    bool broken() const { return broken_; }   // a write failed: the client went away
    /// Non-blocking check that the client closed its side of the connection (for long non-streamed requests).
    bool client_closed();

private:
    int fd_;
    bool sent_ = false, broken_ = false;
    bool write_all(const std::string& data);
};

using Handler = std::function<void(const Request&, Responder&)>;

/// Serve forever on host:port (returns false if the socket cannot be bound). At most `max_connections` are handled
/// at once; further connections get 503.
bool serve(const std::string& host, int port, const Handler& handler, size_t max_body = 16u << 20,
           int max_connections = 64);

} // namespace http
} // namespace q38rt
