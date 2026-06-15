#ifndef __PERF_H_CA__
#define __PERF_H_CA__

#include <All.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace Kernel {
namespace perf {

using PerfClock = std::chrono::steady_clock;
using PerfDuration = std::chrono::nanoseconds;

struct PerfConfig {
    UINT warmup_iters;
    UINT measure_iters;
    BOOL sort_samples;

    PerfConfig(UINT warmup = 10,
               UINT measure = 100,
               BOOL sort = true)
        : warmup_iters(warmup),
          measure_iters(measure),
          sort_samples(sort) {}
};

struct PerfStats {
    std::string name;
    UINT warmup_iters;
    UINT measure_iters;
    uint64_t total_ns;
    uint64_t min_ns;
    uint64_t max_ns;
    double mean_ns;
    double p50_ns;
    double p95_ns;
    std::vector<uint64_t> samples_ns;

    PerfStats()
        : name(),
          warmup_iters(0),
          measure_iters(0),
          total_ns(0),
          min_ns(0),
          max_ns(0),
          mean_ns(0.0),
          p50_ns(0.0),
          p95_ns(0.0),
          samples_ns() {}
};

class ScopeTimer {
public:
    explicit ScopeTimer(uint64_t* sink_ns)
        : _sink_ns(sink_ns),
          _start(PerfClock::now()) {}

    ~ScopeTimer() {
        if (nullptr != _sink_ns) {
            const auto end = PerfClock::now();
            *_sink_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<PerfDuration>(end - _start).count()
            );
        }
    }

private:
    uint64_t* _sink_ns;
    PerfClock::time_point _start;
};

inline double ns_to_us(double ns) {
    return ns / 1000.0;
}

inline double ns_to_ms(double ns) {
    return ns / 1000.0 / 1000.0;
}

inline double percentile_sorted(const std::vector<uint64_t>& samples, double q) {
    if (samples.empty()) {
        return 0.0;
    }
    if (q <= 0.0) {
        return static_cast<double>(samples.front());
    }
    if (q >= 1.0) {
        return static_cast<double>(samples.back());
    }

    const double pos = q * static_cast<double>(samples.size() - 1);
    const size_t lo = static_cast<size_t>(pos);
    const size_t hi = std::min(lo + 1, samples.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return static_cast<double>(samples[lo]) * (1.0 - frac)
        + static_cast<double>(samples[hi]) * frac;
}

inline PerfStats make_stats(const std::string& name,
                            const PerfConfig& cfg,
                            std::vector<uint64_t> samples_ns) {
    PerfStats stats;
    stats.name = name;
    stats.warmup_iters = cfg.warmup_iters;
    stats.measure_iters = cfg.measure_iters;
    stats.samples_ns = std::move(samples_ns);

    if (stats.samples_ns.empty()) {
        return stats;
    }

    if (cfg.sort_samples) {
        std::sort(stats.samples_ns.begin(), stats.samples_ns.end());
    }

    stats.min_ns = stats.samples_ns.front();
    stats.max_ns = stats.samples_ns.back();
    stats.total_ns = 0;
    for (auto it = stats.samples_ns.begin(); it != stats.samples_ns.end(); ++it) {
        stats.total_ns += *it;
    }
    stats.mean_ns = static_cast<double>(stats.total_ns)
        / static_cast<double>(stats.samples_ns.size());
    stats.p50_ns = percentile_sorted(stats.samples_ns, 0.50);
    stats.p95_ns = percentile_sorted(stats.samples_ns, 0.95);
    return stats;
}

template <typename Fn>
PerfStats run(const std::string& name,
              Fn&& fn,
              const PerfConfig& cfg = PerfConfig()) {
    for (UINT i = 0; i < cfg.warmup_iters; ++i) {
        fn();
    }

    std::vector<uint64_t> samples_ns;
    samples_ns.reserve(cfg.measure_iters);
    for (UINT i = 0; i < cfg.measure_iters; ++i) {
        const auto begin = PerfClock::now();
        fn();
        const auto end = PerfClock::now();
        samples_ns.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<PerfDuration>(end - begin).count()
        ));
    }

    return make_stats(name, cfg, std::move(samples_ns));
}

template <typename SetupFn, typename RunFn>
PerfStats run_with_setup(const std::string& name,
                         SetupFn&& setup_fn,
                         RunFn&& run_fn,
                         const PerfConfig& cfg = PerfConfig()) {
    for (UINT i = 0; i < cfg.warmup_iters; ++i) {
        setup_fn();
        run_fn();
    }

    std::vector<uint64_t> samples_ns;
    samples_ns.reserve(cfg.measure_iters);
    for (UINT i = 0; i < cfg.measure_iters; ++i) {
        setup_fn();
        const auto begin = PerfClock::now();
        run_fn();
        const auto end = PerfClock::now();
        samples_ns.push_back(static_cast<uint64_t>(
            std::chrono::duration_cast<PerfDuration>(end - begin).count()
        ));
    }

    return make_stats(name, cfg, std::move(samples_ns));
}

inline std::string format_stats(const PerfStats& stats) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "[perf] " << stats.name
        << " warmup=" << stats.warmup_iters
        << " measure=" << stats.measure_iters
        << " mean_us=" << ns_to_us(stats.mean_ns)
        << " p50_us=" << ns_to_us(stats.p50_ns)
        << " p95_us=" << ns_to_us(stats.p95_ns)
        << " min_us=" << ns_to_us(static_cast<double>(stats.min_ns))
        << " max_us=" << ns_to_us(static_cast<double>(stats.max_ns));
    return oss.str();
}

inline double speedup(const PerfStats& baseline, const PerfStats& current) {
    if (current.mean_ns <= 0.0) {
        return 0.0;
    }
    return baseline.mean_ns / current.mean_ns;
}

inline std::string format_comparison(const PerfStats& baseline,
                                     const PerfStats& current) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "[perf] " << current.name
        << " baseline_mean_us=" << ns_to_us(baseline.mean_ns)
        << " current_mean_us=" << ns_to_us(current.mean_ns)
        << " speedup=" << speedup(baseline, current) << "x";
    return oss.str();
}

} // namespace perf
} // namespace Kernel

#endif
