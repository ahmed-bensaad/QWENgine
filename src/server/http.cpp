/// @file http.cpp
/// See http.h.

#include "server/http.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>

namespace q38rt {
namespace http {

namespace {

const char* reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

const char* kCors =
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

// Read one request. Returns 0 on success, an HTTP status on a malformed request, -1 if the connection closed.
int read_request(int fd, Request& req, size_t max_body) {
    std::string buf;
    char tmp[8192];
    size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) {
        const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return -1;
        buf.append(tmp, (size_t)n);
        hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos && buf.size() > (64u << 10)) return 400;
    }
    // request line
    const size_t l1 = buf.find("\r\n");
    const std::string line = buf.substr(0, l1);
    const size_t s1 = line.find(' '), s2 = line.find(' ', s1 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos) return 400;
    req.method = line.substr(0, s1);
    req.path = line.substr(s1 + 1, s2 - s1 - 1);
    const size_t q = req.path.find('?');
    if (q != std::string::npos) req.path.resize(q);
    // headers
    size_t pos = l1 + 2;
    while (pos < hdr_end) {
        const size_t e = buf.find("\r\n", pos);
        const std::string h = buf.substr(pos, e - pos);
        pos = e + 2;
        const size_t c = h.find(':');
        if (c == std::string::npos) continue;
        std::string name = h.substr(0, c), value = h.substr(c + 1);
        for (char& ch : name) ch = (char)tolower((unsigned char)ch);
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
        req.headers[name] = value;
    }
    size_t len = 0;
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        char* end = nullptr;
        const unsigned long long v = strtoull(it->second.c_str(), &end, 10);
        if (!end || *end) return 400;
        len = (size_t)v;
    } else if (req.headers.count("transfer-encoding")) {
        return 400;   // chunked request bodies are not supported
    }
    if (len > max_body) return 413;
    req.body = buf.substr(hdr_end + 4);
    while (req.body.size() < len) {
        const ssize_t n = recv(fd, tmp, std::min(sizeof tmp, len - req.body.size()), 0);
        if (n <= 0) return -1;
        req.body.append(tmp, (size_t)n);
    }
    req.body.resize(len);
    return 0;
}

} // namespace

bool Responder::write_all(const std::string& data) {
    if (broken_) return false;
    size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) { broken_ = true; return false; }
        off += (size_t)n;
    }
    return true;
}

bool Responder::send(int status, const std::string& content_type, const std::string& body) {
    if (sent_) return false;
    sent_ = true;
    std::string h = "HTTP/1.1 " + std::to_string(status) + " " + reason(status) + "\r\n";
    if (!content_type.empty()) h += "Content-Type: " + content_type + "\r\n";
    h += "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n";
    h += kCors;
    h += "\r\n";
    return write_all(h + body);
}

bool Responder::start_events() {
    if (sent_) return false;
    sent_ = true;
    std::string h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n";
    h += kCors;
    h += "\r\n";
    return write_all(h);
}

bool Responder::event(const std::string& payload) { return write_all("data: " + payload + "\n\n"); }

bool Responder::event(const std::string& name, const std::string& payload) {
    return write_all("event: " + name + "\ndata: " + payload + "\n\n");
}

bool Responder::client_closed() {
    if (broken_) return true;
    char c;
    const ssize_t n = recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) broken_ = true;
    return broken_;
}

namespace {

void handle_connection(int cfd, const Handler& handler, size_t max_body) {
    Request req;
    Responder res(cfd);
    const int rc = read_request(cfd, req, max_body);
    if (rc > 0) {
        res.send(rc, "application/json", "{\"error\":{\"message\":\"malformed request\"}}");
    } else if (rc == 0) {
        if (req.method == "OPTIONS") res.send(204, "", "");
        else handler(req, res);
        if (!res.sent()) res.send(500, "application/json", "{\"error\":{\"message\":\"no response\"}}");
    }
    close(cfd);
}

} // namespace

bool serve(const std::string& host, int port, const Handler& handler, size_t max_body, int max_connections) {
    signal(SIGPIPE, SIG_IGN);
    const int sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); return false; }
    int one = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) { fprintf(stderr, "bad host %s\n", host.c_str()); close(sfd); return false; }
    if (bind(sfd, (sockaddr*)&addr, sizeof addr) < 0 || listen(sfd, 16) < 0) { perror("bind/listen"); close(sfd); return false; }
    static std::atomic<int> active{0};
    while (true) {
        const int cfd = accept(sfd, nullptr, nullptr);
        if (cfd < 0) continue;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (active.load() >= max_connections) {
            Responder(cfd).send(503, "application/json", "{\"error\":{\"message\":\"too many connections\"}}");
            close(cfd);
            continue;
        }
        active++;
        std::thread([cfd, &handler, max_body]() {
            handle_connection(cfd, handler, max_body);
            active--;
        }).detach();
    }
}

} // namespace http
} // namespace q38rt
