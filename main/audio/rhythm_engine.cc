#include "rhythm_engine.h"

#include <cmath>
#include <cstring>

#include <esp_log.h>
#include <esp_timer.h>

namespace {

const char* TAG = "Rhythm";

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinFrequency = 40.0f;
constexpr float kDbFloor = -60.0f;
constexpr float kDbCeiling = -15.0f;
constexpr float kMicGain = 2.4f;
constexpr float kSignalThreshold = 0.025f;
constexpr uint32_t kSilenceHoldMs = 400;
constexpr float kLoudLevel = 0.05f;
constexpr float kRiseSmoothing = 0.35f;
constexpr float kDecaySmoothing = 0.65f;

// 鼓点：低频前 12 段（约 40~370Hz）的谱通量
constexpr float kFluxAverageSmoothing = 0.05f;
constexpr float kFluxTriggerFactor = 1.6f;
constexpr float kFluxMin = 0.08f;
constexpr float kBeatMinLevel = 0.06f;
constexpr float kBeatMinLow = 0.24f;
constexpr uint32_t kBeatMinGapMs = 240;
constexpr float kTempoBandLow = 0.75f;
constexpr float kTempoBandHigh = 1.33f;
constexpr uint8_t kTempoResyncCount = 3;

}  // namespace

RhythmEngine& RhythmEngine::GetInstance() {
    static RhythmEngine instance;
    return instance;
}

uint32_t RhythmEngine::NowMs() {
    return static_cast<uint32_t>(esp_timer_get_time() / 1000ULL);
}

void RhythmEngine::BuildTables() {
    if (tables_ready_) {
        return;
    }

    // 32 个等比频段：40Hz -> 8kHz（16kHz 采样的奈奎斯特频率）
    constexpr float kSampleRate = 16000.0f;
    const float nyquist = kSampleRate / 2.0f;
    const float top = nyquist > kMinFrequency * 2.0f ? nyquist : kMinFrequency * 2.0f;
    for (size_t index = 0; index <= kBarCount; ++index) {
        const float ratio = static_cast<float>(index) / static_cast<float>(kBarCount);
        const float frequency = kMinFrequency * powf(top / kMinFrequency, ratio);
        int bin = static_cast<int>(lroundf(frequency * kFftSize / kSampleRate));
        if (bin < 1) {
            bin = 1;
        }
        if (bin > static_cast<int>(kFftSize / 2 - 1)) {
            bin = static_cast<int>(kFftSize / 2 - 1);
        }
        if (index > 0 && bin <= band_edge_[index - 1]) {
            bin = band_edge_[index - 1] + 1;
        }
        band_edge_[index] = static_cast<uint16_t>(bin);
    }

    // Hann 窗
    for (size_t index = 0; index < kFftSize; ++index) {
        window_[index] = 0.5f - 0.5f * cosf(2.0f * kPi * static_cast<float>(index) /
                                           static_cast<float>(kFftSize - 1));
    }

    // 旋转因子 e^{-2πi k/N}，只需要前一半
    for (size_t index = 0; index < kFftSize / 2; ++index) {
        const float angle = -2.0f * kPi * static_cast<float>(index) / static_cast<float>(kFftSize);
        twiddle_re_[index] = cosf(angle);
        twiddle_im_[index] = sinf(angle);
    }

    tables_ready_ = true;
}

void RhythmEngine::Start() {
    // 先停掉，避免音频任务在重置参数时还往半初始化的状态里喂数据。
    running_.store(false, std::memory_order_release);
    BuildTables();

    block_fill_ = 0;
    memset(bars_, 0, sizeof(bars_));
    memset(raw_bands_, 0, sizeof(raw_bands_));
    memset(previous_raw_, 0, sizeof(previous_raw_));
    avg_level_ = 0.0f;
    flux_average_ = 0.0f;
    last_beat_ms_ = 0;
    for (size_t index = 0; index < kIntervalCount; ++index) {
        intervals_[index] = 500;
    }
    interval_cursor_ = 0;
    beat_period_ms_ = 0;
    out_of_band_count_ = 0;

    const uint32_t now = NowMs();
    last_signal_ms_ = now;
    last_loud_ms_.store(now, std::memory_order_relaxed);
    loud_since_ms_.store(0, std::memory_order_relaxed);
    loud_.store(false, std::memory_order_relaxed);

    beat_pending_.store(false, std::memory_order_relaxed);
    last_beat_strength_.store(1.0f, std::memory_order_relaxed);
    beat_period_ms_atomic_.store(0, std::memory_order_relaxed);
    beat_count_.store(0, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        memset(shared_bars_, 0, sizeof(shared_bars_));
        shared_avg_level_ = 0.0f;
        shared_flux_ = 0.0f;
        shared_threshold_ = 0.0f;
        shared_low_ = 0.0f;
    }

    running_.store(true, std::memory_order_release);
    ESP_LOGI(TAG, "started (%u bars, %u-point FFT)", static_cast<unsigned>(kBarCount),
             static_cast<unsigned>(kFftSize));
}

void RhythmEngine::Stop() {
    running_.store(false, std::memory_order_release);
    block_fill_ = 0;
    beat_pending_.store(false, std::memory_order_relaxed);
    loud_.store(false, std::memory_order_relaxed);
    loud_since_ms_.store(0, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        memset(shared_bars_, 0, sizeof(shared_bars_));
        shared_avg_level_ = 0.0f;
        shared_flux_ = 0.0f;
        shared_threshold_ = 0.0f;
        shared_low_ = 0.0f;
    }

    ESP_LOGI(TAG, "stopped (beats=%u)", BeatCount());
}

void RhythmEngine::Feed(const int16_t* data, size_t samples, int channels) {
    if (!running_.load(std::memory_order_acquire) || data == nullptr || samples == 0 || channels <= 0) {
        return;
    }

    const size_t frames = samples / static_cast<size_t>(channels);
    for (size_t index = 0; index < frames; ++index) {
        // 用第一个声道：和唤醒词保持一致，也避免左右麦克风相位相消。
        block_[block_fill_++] = data[index * static_cast<size_t>(channels)];
        if (block_fill_ >= kFftSize) {
            block_fill_ = 0;
            ProcessBlock(NowMs());
        }
    }
}

void RhythmEngine::ProcessBlock(uint32_t now_ms) {
    float peak = 0.0f;
    for (size_t index = 0; index < kFftSize; ++index) {
        const float value = static_cast<float>(block_[index]) / 32768.0f;
        const float magnitude = fabsf(value);
        if (magnitude > peak) {
            peak = magnitude;
        }
        re_[index] = value * kMicGain * window_[index];
        im_[index] = 0.0f;
    }
    if (peak > kSignalThreshold) {
        last_signal_ms_ = now_ms;
    }

    Fft();
    ComputeSpectrum(now_ms);
}

void RhythmEngine::Fft() {
    // 位反转置换
    for (uint16_t index = 1, reversed = 0; index < kFftSize; ++index) {
        uint16_t bit = kFftSize >> 1;
        for (; reversed & bit; bit >>= 1) {
            reversed ^= bit;
        }
        reversed ^= bit;
        if (index < reversed) {
            const float tr = re_[index];
            re_[index] = re_[reversed];
            re_[reversed] = tr;
            const float ti = im_[index];
            im_[index] = im_[reversed];
            im_[reversed] = ti;
        }
    }

    // 迭代 radix-2
    for (uint16_t len = 2; len <= kFftSize; len <<= 1) {
        const uint16_t half = len >> 1;
        const uint16_t step = kFftSize / len;
        for (uint16_t base = 0; base < kFftSize; base += len) {
            for (uint16_t k = 0; k < half; ++k) {
                const float wr = twiddle_re_[k * step];
                const float wi = twiddle_im_[k * step];
                const uint16_t a = base + k;
                const uint16_t b = base + k + half;
                const float tr = re_[b] * wr - im_[b] * wi;
                const float ti = re_[b] * wi + im_[b] * wr;
                re_[b] = re_[a] - tr;
                im_[b] = im_[a] - ti;
                re_[a] += tr;
                im_[a] += ti;
            }
        }
    }
}

void RhythmEngine::ComputeSpectrum(uint32_t now_ms) {
    const float range = kDbCeiling - kDbFloor > 1.0f ? kDbCeiling - kDbFloor : 10.0f;
    const float scale = 2.0f / static_cast<float>(kFftSize);
    const bool silent = (now_ms - last_signal_ms_) > kSilenceHoldMs;
    const size_t half = kFftSize / 2;

    for (size_t bar = 0; bar < kBarCount; ++bar) {
        uint16_t start = band_edge_[bar];
        uint16_t end = band_edge_[bar + 1];
        if (end > half) {
            end = static_cast<uint16_t>(half);
        }
        float level = kDbFloor;
        for (uint16_t bin = start; bin < end; ++bin) {
            const float magnitude = sqrtf(re_[bin] * re_[bin] + im_[bin] * im_[bin]) * scale;
            const float db = 20.0f * log10f(magnitude > 1e-7f ? magnitude : 1e-7f);
            if (db > level) {
                level = db;
            }
        }

        float value = (level - kDbFloor) / range;
        if (value < 0.0f) {
            value = 0.0f;
        }
        if (value > 1.0f) {
            value = 1.0f;
        }
        if (silent) {
            value = 0.0f;
        }

        raw_bands_[bar] = value;
        const float diff = value - bars_[bar];
        bars_[bar] += diff * (diff >= 0.0f ? kRiseSmoothing : kDecaySmoothing);
    }

    float sum = 0.0f;
    for (size_t bar = 0; bar < kBarCount; ++bar) {
        sum += bars_[bar];
    }
    avg_level_ = sum / static_cast<float>(kBarCount);

    // 低频谱通量：相邻两帧的正向增量
    float flux = 0.0f;
    float low_max = 0.0f;
    for (size_t bar = 0; bar < kBeatBands && bar < kBarCount; ++bar) {
        const float delta = raw_bands_[bar] - previous_raw_[bar];
        if (delta > 0.0f) {
            flux += delta;
        }
        previous_raw_[bar] = raw_bands_[bar];
        if (raw_bands_[bar] > low_max) {
            low_max = raw_bands_[bar];
        }
    }

    flux_average_ += (flux - flux_average_) * kFluxAverageSmoothing;
    const float threshold = fmaxf(kFluxMin, flux_average_ * kFluxTriggerFactor);
    const uint32_t min_gap = beat_period_ms_
                                 ? static_cast<uint32_t>(fmaxf(200.0f, static_cast<float>(beat_period_ms_) * 0.5f))
                                 : kBeatMinGapMs;

    const bool gap_ok = (last_beat_ms_ == 0) || ((now_ms - last_beat_ms_) >= min_gap);
    if (flux >= threshold && avg_level_ > kBeatMinLevel && low_max >= kBeatMinLow && gap_ok) {
        const uint32_t interval = last_beat_ms_ ? (now_ms - last_beat_ms_) : 0;
        bool accept = true;
        if (last_beat_ms_ != 0 && interval > 0 && interval < 2000) {
            if (beat_period_ms_ != 0) {
                const uint32_t low_edge = static_cast<uint32_t>(beat_period_ms_ * kTempoBandLow);
                const uint32_t high_edge = static_cast<uint32_t>(beat_period_ms_ * kTempoBandHigh);
                const bool near_half = interval * 5 >= beat_period_ms_ * 2 && interval * 5 <= beat_period_ms_ * 3;
                const bool near_double = interval * 10 >= beat_period_ms_ * 17 && interval * 10 <= beat_period_ms_ * 23;
                if (interval >= low_edge && interval <= high_edge) {
                    out_of_band_count_ = 0;
                } else if (near_half || near_double) {
                    // 典型的漏拍/多拍：只点头，不改周期估计
                    accept = false;
                } else {
                    accept = false;
                    if (++out_of_band_count_ >= kTempoResyncCount) {
                        out_of_band_count_ = 0;
                        for (size_t index = 0; index < kIntervalCount; ++index) {
                            intervals_[index] = interval;
                        }
                        beat_period_ms_ = interval;
                    }
                }
            }

            if (accept) {
                intervals_[interval_cursor_] = interval;
                interval_cursor_ = static_cast<uint8_t>((interval_cursor_ + 1) % kIntervalCount);

                uint32_t sorted[kIntervalCount];
                for (size_t index = 0; index < kIntervalCount; ++index) {
                    sorted[index] = intervals_[index];
                }
                for (size_t a = 1; a < kIntervalCount; ++a) {
                    const uint32_t key = sorted[a];
                    int b = static_cast<int>(a) - 1;
                    while (b >= 0 && sorted[b] > key) {
                        sorted[b + 1] = sorted[b];
                        --b;
                    }
                    sorted[b + 1] = key;
                }
                beat_period_ms_ = sorted[kIntervalCount / 2];
                beat_period_ms_atomic_.store(beat_period_ms_, std::memory_order_relaxed);
            }
        }

        last_beat_ms_ = now_ms;
        beat_pending_.store(true, std::memory_order_release);
        const float strength = flux / threshold;
        last_beat_strength_.store(strength, std::memory_order_relaxed);
        const uint32_t count = beat_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        // 逐拍日志放 DEBUG：串口打印会占用音频输入任务的时间，默认不刷屏。
        ESP_LOGD(TAG, "#%u x%.1f bpm=%u", count, strength,
                 beat_period_ms_ ? (unsigned)(60000UL / beat_period_ms_) : 0U);
    }

    {
        std::lock_guard<std::mutex> lock(snapshot_mutex_);
        memcpy(shared_bars_, bars_, sizeof(shared_bars_));
        shared_avg_level_ = avg_level_;
        shared_flux_ = flux;
        shared_threshold_ = threshold;
        shared_low_ = low_max;
    }

    const bool loud_now = avg_level_ > kLoudLevel || (now_ms - last_signal_ms_) <= kSilenceHoldMs;
    loud_.store(loud_now, std::memory_order_relaxed);
    if (loud_now) {
        if (loud_since_ms_.load(std::memory_order_relaxed) == 0) {
            loud_since_ms_.store(now_ms, std::memory_order_relaxed);
        }
        last_loud_ms_.store(now_ms, std::memory_order_relaxed);
    } else {
        loud_since_ms_.store(0, std::memory_order_relaxed);
    }
}

void RhythmEngine::GetBars(float* out, size_t count) const {
    if (out == nullptr || count == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    const size_t limit = count < kBarCount ? count : kBarCount;
    for (size_t index = 0; index < limit; ++index) {
        out[index] = shared_bars_[index];
    }
    for (size_t index = limit; index < count; ++index) {
        out[index] = 0.0f;
    }
}

float RhythmEngine::AverageLevel() const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    return shared_avg_level_;
}

uint32_t RhythmEngine::LoudDurationMs(uint32_t now_ms) const {
    if (!loud_.load(std::memory_order_relaxed)) {
        return 0;
    }
    const uint32_t since = loud_since_ms_.load(std::memory_order_relaxed);
    return since ? (now_ms - since) : 0;
}

uint32_t RhythmEngine::SilenceMs(uint32_t now_ms) const {
    const uint32_t last = last_loud_ms_.load(std::memory_order_relaxed);
    return last ? (now_ms - last) : 0;
}

bool RhythmEngine::ConsumeBeat() {
    return beat_pending_.exchange(false, std::memory_order_acq_rel);
}

void RhythmEngine::GetDiagnostics(Diagnostics& out) const {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    out.flux = shared_flux_;
    out.threshold = shared_threshold_;
    out.low = shared_low_;
    out.level = shared_avg_level_;
    const uint32_t period = beat_period_ms_atomic_.load(std::memory_order_relaxed);
    out.bpm = period ? static_cast<uint32_t>(60000UL / period) : 0U;
    out.beats = beat_count_.load(std::memory_order_relaxed);
}
