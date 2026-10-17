#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

/**
 * 音乐律动的听音分析。
 *
 * 输入：音频输入任务复制过来的 16kHz 麦克风数据（单声道或双声道交错）。
 * 输出：
 *   - 32 个频段的能量（给屏幕底部的频谱条）
 *   - 低频鼓点（每检测到一拍，ConsumeBeat() 返回一次 true）
 *   - 节拍周期（用于选择点头幅度/速度）
 *   - "是否有声音"和连续有声时长（用于笑脸和自动退出）
 *
 * 只在设备本地的音频输入任务里跑，不联网、不依赖服务器。
 * 算法参考 Finch 的 finch-chan，但按小智固件的接口重写，不引入第三方库。
 */
class RhythmEngine {
public:
    static RhythmEngine& GetInstance();

    void Start();
    void Stop();
    bool IsRunning() const { return running_.load(std::memory_order_acquire); }

    /** 音频输入任务每 10ms 调一次。samples 是 int16 的个数（含声道）。 */
    void Feed(const int16_t* data, size_t samples, int channels);

    /** 拷贝当前 32 个频段（0..1）。 */
    void GetBars(float* out, size_t count) const;
    float AverageLevel() const;
    bool IsLoud() const { return loud_.load(std::memory_order_relaxed); }
    uint32_t LoudDurationMs(uint32_t now_ms) const;
    uint32_t SilenceMs(uint32_t now_ms) const;

    /** 读一次就清：这一帧刚踩到一拍。 */
    bool ConsumeBeat();
    float LastBeatStrength() const { return last_beat_strength_.load(std::memory_order_relaxed); }
    uint32_t BeatPeriodMs() const { return beat_period_ms_atomic_.load(std::memory_order_relaxed); }
    uint32_t BeatCount() const { return beat_count_.load(std::memory_order_relaxed); }

    /** 调参用的诊断快照；由控制定时器打印，避免在音频任务里打串口。 */
    struct Diagnostics {
        float flux = 0.0f;
        float threshold = 0.0f;
        float low = 0.0f;
        float level = 0.0f;
        uint32_t bpm = 0;
        uint32_t beats = 0;
    };
    void GetDiagnostics(Diagnostics& out) const;

private:
    RhythmEngine() = default;

    void BuildTables();
    void ProcessBlock(uint32_t now_ms);
    void ComputeSpectrum(uint32_t now_ms);
    void Fft();
    static uint32_t NowMs();

    static constexpr size_t kFftSize = 512;
    static constexpr size_t kBarCount = 32;
    static constexpr size_t kBeatBands = 12;
    static constexpr size_t kIntervalCount = 5;

    std::atomic<bool> running_{false};

    // 以下只在音频输入任务里读写
    int16_t block_[kFftSize] = {};
    size_t block_fill_ = 0;
    float re_[kFftSize] = {};
    float im_[kFftSize] = {};
    float window_[kFftSize] = {};
    float twiddle_re_[kFftSize / 2] = {};
    float twiddle_im_[kFftSize / 2] = {};
    uint16_t band_edge_[kBarCount + 1] = {};
    float bars_[kBarCount] = {};
    float raw_bands_[kBarCount] = {};
    float previous_raw_[kBarCount] = {};
    float avg_level_ = 0.0f;
    float flux_average_ = 0.0f;
    uint32_t last_signal_ms_ = 0;
    uint32_t last_beat_ms_ = 0;
    uint32_t intervals_[kIntervalCount] = {500, 500, 500, 500, 500};
    uint8_t interval_cursor_ = 0;
    uint32_t beat_period_ms_ = 0;
    uint8_t out_of_band_count_ = 0;
    bool tables_ready_ = false;

    // 给界面/诊断读的快照（音频任务写、其他任务读）
    mutable std::mutex snapshot_mutex_;
    float shared_bars_[kBarCount] = {};
    float shared_avg_level_ = 0.0f;
    float shared_flux_ = 0.0f;
    float shared_threshold_ = 0.0f;
    float shared_low_ = 0.0f;

    // 跨任务状态
    std::atomic<bool> beat_pending_{false};
    std::atomic<float> last_beat_strength_{1.0f};
    std::atomic<uint32_t> beat_period_ms_atomic_{0};
    std::atomic<uint32_t> beat_count_{0};
    std::atomic<bool> loud_{false};
    std::atomic<uint32_t> loud_since_ms_{0};
    std::atomic<uint32_t> last_loud_ms_{0};
};
