#pragma once

#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <ctime>
#include <memory>
#include <string>

namespace gs_native {

inline std::string FormatLogScope(std::string name) {
    name = name.substr(0, name.find('('));
    name = name.substr(name.rfind(' ') + 1);
    for (const char* token : {"`anonymous-namespace'::", "{anonymous}::", "gs_native::"}) {
        size_t pos = name.find(token);
        if (pos != std::string::npos) {
            name.erase(pos, std::strlen(token));
        }
    }

    size_t pos = name.rfind("::");
    if (pos == std::string::npos) {
        return "[" + name + "] ";
    }
    return "[" + name.substr(0, pos) + "][" + name.substr(pos + 2) + "] ";
}

class LogScopeFormatter : public spdlog::custom_flag_formatter {
public:
    void format(
        const spdlog::details::log_msg& msg,
        const std::tm&,
        spdlog::memory_buf_t& dest) override {
        if (msg.source.empty()) {
            return;
        }
        std::string scope = FormatLogScope(msg.source.funcname);
        dest.append(scope.data(), scope.data() + scope.size());
    }

    std::unique_ptr<spdlog::custom_flag_formatter> clone() const override {
        return std::make_unique<LogScopeFormatter>();
    }
};

inline spdlog::logger* GetLogInstance() {
    static spdlog::logger* logger = [] {
        auto formatter = std::make_unique<spdlog::pattern_formatter>();
        formatter->add_flag<LogScopeFormatter>('*').set_pattern("[%H:%M:%S.%e] [%l] [%s:%#] %*%v");
        spdlog::set_formatter(std::move(formatter));
        return spdlog::default_logger_raw();
    }();
    return logger;
}

}

#define GS_NATIVE_INFO(...) SPDLOG_LOGGER_INFO(gs_native::GetLogInstance(), __VA_ARGS__)
#define GS_NATIVE_WARN(...) SPDLOG_LOGGER_WARN(gs_native::GetLogInstance(), __VA_ARGS__)
#define GS_NATIVE_ERROR(...) SPDLOG_LOGGER_ERROR(gs_native::GetLogInstance(), __VA_ARGS__)
