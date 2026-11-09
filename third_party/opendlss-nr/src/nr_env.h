// The network's NEURAL_FORGE_* variables, read with the rules the rest of Neural Forge reads its variables by
// (neural_forge_protocol::env): the value is trimmed of surrounding whitespace (a launch option written by hand
// often carries a stray space), and an empty one is unset. A switch is on only for exactly "1" and off for "0";
// a number is a whole decimal number that fits 32 bits. Anything else is said once on stderr and ignored: the
// default applies, as if the variable were unset.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <string>

namespace nr::env {

// The variable's value, trimmed; nullopt when it is unset or blank.
inline std::optional<std::string> value(const char* name) {
  const char* raw = std::getenv(name);
  if (!raw) return std::nullopt;
  const std::string text(raw);
  constexpr const char* kSpace = " \t\n\v\f\r";
  const size_t first = text.find_first_not_of(kSpace);
  if (first == std::string::npos) return std::nullopt;
  return text.substr(first, text.find_last_not_of(kSpace) - first + 1);
}

// Says, once per variable, that its value is not what the variable takes and is ignored.
inline void ignored(const char* name, const std::string& value, const char* takes) {
  static std::mutex lock;
  static std::set<std::string> said;
  std::lock_guard<std::mutex> guard(lock);
  if (!said.insert(name).second) return;
  fprintf(stderr, "[neural-forge-layer] [native] %s=\"%s\" ignored: it takes %s\n", name, value.c_str(), takes);
}

// A switch: "1" on, "0" off, `fallback` when unset (or ignored).
inline bool flag(const char* name, bool fallback) {
  const std::optional<std::string> v = value(name);
  if (!v) return fallback;
  if (*v == "1") return true;
  if (*v == "0") return false;
  ignored(name, *v, "0 or 1");
  return fallback;
}

// A whole decimal number of 32 bits; nullopt when unset (or ignored).
inline std::optional<uint32_t> number(const char* name) {
  const std::optional<std::string> v = value(name);
  if (!v) return std::nullopt;
  // Digits only: strtoul alone takes a sign and leading spaces, and wraps a negative number.
  if (v->find_first_not_of("0123456789") == std::string::npos) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v->c_str(), &end, 10);
    if (errno == 0 && *end == '\0' && n <= UINT32_MAX) return static_cast<uint32_t>(n);
  }
  ignored(name, *v, "a whole number");
  return std::nullopt;
}

// The number, or `fallback` (which may be outside what the variable takes, as -1 for "automatic").
inline int64_t numberOr(const char* name, int64_t fallback) {
  const std::optional<uint32_t> n = number(name);
  return n ? int64_t(*n) : fallback;
}

}  // namespace nr::env
