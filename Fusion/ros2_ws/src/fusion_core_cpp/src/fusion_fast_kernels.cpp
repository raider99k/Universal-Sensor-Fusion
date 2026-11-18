#include "fusion_core_cpp/fusion_fast_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

inline double clamp_d(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline double read_field_as_double(const uint8_t* ptr, int32_t datatype) {
    switch (datatype) {
        case 1: {  // INT8
            int8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 2: {  // UINT8
            uint8_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 3: {  // INT16
            int16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 4: {  // UINT16
            uint16_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 5: {  // INT32
            int32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 6: {  // UINT32
            uint32_t v = 0;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 7: {  // FLOAT32
            float v = 0.0f;
            std::memcpy(&v, ptr, sizeof(v));
            return static_cast<double>(v);
        }
        case 8: {  // FLOAT64
            double v = 0.0;
            std::memcpy(&v, ptr, sizeof(v));
            return v;
        }
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

struct ZoneCandPt {
    int32_t zone_id;
    int32_t z_quant;
    double x;
    double y;
    double z;
    double r;
};

struct PreRoiPt {
    int32_t orig_idx;
    double x;
    double y;
    double z;
};

}  // namespace

FUSION_CPP_EXPORT int32_t cpp_extract_and_filter_ground(
    const uint8_t* raw_bytes,
    const fusion_core_cpp::LidarPipelineConfigC* cfg,
    double* out_x,
    double* out_y,
    double* out_z,
    double* out_refl,
    int32_t max_out,
    fusion_core_cpp::LidarExtractStatsC* out_stats
) {
    if (!raw_bytes || !cfg || !out_x || !out_y || !out_z || max_out <= 0) {
        return 0;
    }

    const int32_t n_total = cfg->n_total;
    const int32_t point_step = cfg->point_step;
    if (n_total <= 0 || point_step <= 0) {
        if (out_stats) {
            out_stats->raw_points = 0;
            out_stats->kept_points = 0;
            out_stats->self_return_rejects = 0;
        }
        return 0;
    }

    const double syaw = cfg->sensor_yaw_rad;
    const double sx = cfg->sensor_offset_x_m;
    const double sy = cfg->sensor_offset_y_m;
    const double sz = cfg->sensor_offset_z_m;
    const bool use_leveling = (cfg->enable_imu_leveling != 0) &&
                              (std::abs(cfg->sin_pitch) > 1e-6 || std::abs(cfg->sin_roll) > 1e-6);
    const double sin_pitch = use_leveling ? cfg->sin_pitch : 0.0;
    const double sin_roll = use_leveling ? cfg->sin_roll : 0.0;

    const double max_r = std::max({
        std::abs(cfg->roi_x_min),
        std::abs(cfg->roi_x_max),
        std::abs(cfg->roi_y_min),
        std::abs(cfg->roi_y_max)
    });

    double xy_pad = 0.0;
    if (cfg->has_motion) {
        const double max_dt = std::min(cfg->deskew_scan_duration_s, cfg->deskew_max_dt_s);
        xy_pad = (std::abs(cfg->vx) + std::abs(cfg->vy)) * max_dt +
                 std::abs(cfg->wz) * max_dt * max_r + 0.5;
    }
    const double z_pad = use_leveling ? (max_r * (std::abs(sin_pitch) + std::abs(sin_roll)) + 0.05) : 0.0;

    // Fast finite check on strided sample (or full if small)
    int32_t raw_count = 0;
    bool all_sample_finite = (n_total >= 4096);
    if (all_sample_finite) {
        for (int32_t i = 0; i < n_total; i += 64) {
            const uint8_t* p = raw_bytes + static_cast<size_t>(i) * point_step;
            float rx, ry, rz;
            std::memcpy(&rx, p + cfg->off_x, 4);
            std::memcpy(&ry, p + cfg->off_y, 4);
            std::memcpy(&rz, p + cfg->off_z, 4);
            if (!std::isfinite(rx) || !std::isfinite(ry) || !std::isfinite(rz)) {
                all_sample_finite = false;
                break;
            }
        }
    }
    if (all_sample_finite) {
        raw_count = n_total;
    } else {
        for (int32_t i = 0; i < n_total; ++i) {
            const uint8_t* p = raw_bytes + static_cast<size_t>(i) * point_step;
            float rx, ry, rz;
            std::memcpy(&rx, p + cfg->off_x, 4);
            std::memcpy(&ry, p + cfg->off_y, 4);
            std::memcpy(&rz, p + cfg->off_z, 4);
            if (std::isfinite(rx) && std::isfinite(ry) && std::isfinite(rz)) {
                ++raw_count;
            }
        }
    }

    if (raw_count == 0) {
        if (out_stats) {
            out_stats->raw_points = 0;
            out_stats->kept_points = 0;
            out_stats->self_return_rejects = 0;
        }
        return 0;
    }

    // 2D organized ring pre-culling for multi-beam scans (e.g. 64x2048 Ouster)
    int32_t r0_idx = 0;
    int32_t r1_idx = n_total;
    const int32_t h_msg = cfg->height;
    const int32_t w_msg = cfg->width;
    if (h_msg >= 16 && w_msg >= 256 && h_msg * w_msg == n_total) {
        const float sz32 = static_cast<float>(sz);
        const float z_min_g = static_cast<float>(cfg->roi_z_min - z_pad - 0.15);
        const float z_max_g = static_cast<float>(cfg->roi_z_max + z_pad + 0.15);
        int32_t first_row = -1;
        int32_t last_row = -1;
        for (int32_t r = 0; r < h_msg; ++r) {
            bool row_ok = false;
            for (int32_t c = 0; c < w_msg; c += 64) {
                const uint8_t* p = raw_bytes + static_cast<size_t>(r * w_msg + c) * point_step;
                float rz;
                std::memcpy(&rz, p + cfg->off_z, 4);
                const float gz = rz + sz32;
                if (gz >= z_min_g && gz <= z_max_g && gz != sz32) {
                    row_ok = true;
                    break;
                }
            }
            if (row_ok) {
                if (first_row < 0) first_row = r;
                last_row = r;
            }
        }
        if (first_row >= 0) {
            const int32_t r_start = std::max(0, first_row - 1);
            const int32_t r_end = std::min(h_msg, last_row + 2);
            r0_idx = r_start * w_msg;
            r1_idx = r_end * w_msg;
        }
    }

    const float z_lo32 = static_cast<float>(cfg->roi_z_min - sz - z_pad);
    const float z_hi32 = static_cast<float>(cfg->roi_z_max - sz + z_pad);
    const bool has_syaw = std::abs(syaw) > 1e-6;
    const double cs = has_syaw ? std::cos(syaw) : 1.0;
    const double ss = has_syaw ? std::sin(syaw) : 0.0;

    const bool do_sr = (cfg->enable_self_return_suppression != 0);
    const double max_r_xy = std::max(0.0, cfg->self_return_radius_m);
    const double max_r_xy_sq = max_r_xy * max_r_xy;
    const double sr_z_min = cfg->self_return_z_min_m;
    const double sr_z_max = cfg->self_return_z_max_m;
    const double min_r3d = cfg->self_return_min_range_m;
    const double min_r3d_sq = min_r3d * min_r3d;

    const double pre_x_min = cfg->roi_x_min - xy_pad;
    const double pre_x_max = cfg->roi_x_max + xy_pad;
    const double pre_y_min = cfg->roi_y_min - xy_pad;
    const double pre_y_max = cfg->roi_y_max + xy_pad;
    const double pre_z_min = cfg->roi_z_min;
    const double pre_z_max = cfg->roi_z_max;

    thread_local std::vector<PreRoiPt> cand_pts;
    cand_pts.clear();
    if (cand_pts.capacity() < 32768) {
        cand_pts.reserve(32768);
    }

    int32_t self_return_rejects = 0;

    for (int32_t i = r0_idx; i < r1_idx; ++i) {
        const uint8_t* p = raw_bytes + static_cast<size_t>(i) * point_step;
        float rz32;
        std::memcpy(&rz32, p + cfg->off_z, 4);
        if (!(rz32 >= z_lo32 && rz32 <= z_hi32)) {
            continue;
        }
        float rx32, ry32;
        std::memcpy(&rx32, p + cfg->off_x, 4);
        std::memcpy(&ry32, p + cfg->off_y, 4);
        if (!std::isfinite(rx32) || !std::isfinite(ry32)) {
            continue;
        }
        if ((std::abs(sx) > 0.1 || std::abs(sy) > 0.1) && (rx32 * rx32 + ry32 * ry32 <= 1.0f)) {
            ++self_return_rejects;
            continue;
        }
        const double rx = static_cast<double>(rx32);
        const double ry = static_cast<double>(ry32);
        const double rz = static_cast<double>(rz32);
        double xs_d, ys_d;
        if (has_syaw) {
            xs_d = cs * rx - ss * ry + sx;
            ys_d = ss * rx + cs * ry + sy;
        } else {
            xs_d = rx + sx;
            ys_d = ry + sy;
        }
        double zs_d = rz + sz;
        if (use_leveling) {
            zs_d = zs_d + xs_d * sin_pitch - ys_d * sin_roll;
        }
        if (do_sr) {
            const double r_xy_sq = xs_d * xs_d + ys_d * ys_d;
            bool is_sr = (zs_d >= sr_z_min && zs_d <= sr_z_max && r_xy_sq <= max_r_xy_sq);
            if (is_sr && min_r3d > 0.0) {
                is_sr = ((r_xy_sq + zs_d * zs_d) <= min_r3d_sq);
            }
            if (is_sr) {
                ++self_return_rejects;
                continue;
            }
        }
        if (zs_d >= pre_z_min && zs_d <= pre_z_max &&
            xs_d >= pre_x_min && xs_d <= pre_x_max &&
            ys_d >= pre_y_min && ys_d <= pre_y_max) {
            cand_pts.push_back({i, xs_d, ys_d, zs_d});
        }
    }

    if (static_cast<int32_t>(cand_pts.size()) > cfg->max_points_for_clustering && cfg->max_points_for_clustering > 0) {
        const int32_t stride = static_cast<int32_t>(std::ceil(
            static_cast<double>(cand_pts.size()) / static_cast<double>(cfg->max_points_for_clustering)
        ));
        size_t w_idx = 0;
        for (size_t r_idx = 0; r_idx < cand_pts.size(); r_idx += stride) {
            cand_pts[w_idx++] = cand_pts[r_idx];
        }
        cand_pts.resize(w_idx);
    }

    // Continuous motion deskewing on surviving ROI candidate points
    if (cfg->has_motion == 2 && !cand_pts.empty() && cfg->off_t >= 0) {
        const double vx = cfg->vx;
        const double vy = cfg->vy;
        const double wz = cfg->wz;
        for (auto& pt : cand_pts) {
            const uint8_t* p = raw_bytes + static_cast<size_t>(pt.orig_idx) * point_step + cfg->off_t;
            const double pt_ns = read_field_as_double(p, cfg->t_datatype);
            const double pt_s = pt_ns * 1e-9;
            const double dth = wz * pt_s;
            const double cd = 1.0 - 0.5 * (dth * dth);
            const double nx = cd * pt.x - dth * pt.y + vx * pt_s;
            const double ny = dth * pt.x + cd * pt.y + vy * pt_s;
            pt.x = nx;
            pt.y = ny;
        }
    } else if (cfg->has_motion && !cand_pts.empty()) {
        double t_min = std::numeric_limits<double>::infinity();
        double t_max = -std::numeric_limits<double>::infinity();
        bool has_valid_t = false;

        if (cfg->off_t >= 0 && cfg->t_datatype > 0) {
            for (const auto& pt : cand_pts) {
                const uint8_t* p = raw_bytes + static_cast<size_t>(pt.orig_idx) * point_step + cfg->off_t;
                const double tv = read_field_as_double(p, cfg->t_datatype);
                if (std::isfinite(tv)) {
                    if (tv < t_min) t_min = tv;
                    if (tv > t_max) t_max = tv;
                    has_valid_t = true;
                }
            }
            const int32_t smp_step = std::max(1, n_total / 2048);
            for (int32_t i = 0; i < n_total; i += smp_step) {
                const uint8_t* p = raw_bytes + static_cast<size_t>(i) * point_step + cfg->off_t;
                const double tv = read_field_as_double(p, cfg->t_datatype);
                if (std::isfinite(tv)) {
                    if (tv < t_min) t_min = tv;
                    if (tv > t_max) t_max = tv;
                }
            }
        }

        const bool use_t_field = has_valid_t && (t_max > t_min + 1e-9);
        const double inv_t_span = use_t_field ? (1.0 / (t_max - t_min)) : 0.0;
        const double vx = cfg->vx;
        const double vy = cfg->vy;
        const double wz = cfg->wz;
        const double scan_dur = cfg->deskew_scan_duration_s;
        const double max_dt = cfg->deskew_max_dt_s;
        const bool rev_spin = (cfg->deskew_spin_direction < 0.0);

        for (auto& pt : cand_pts) {
            double phase = std::numeric_limits<double>::quiet_NaN();
            if (use_t_field) {
                const uint8_t* p = raw_bytes + static_cast<size_t>(pt.orig_idx) * point_step + cfg->off_t;
                const double tv = read_field_as_double(p, cfg->t_datatype);
                if (std::isfinite(tv)) {
                    phase = clamp_d((tv - t_min) * inv_t_span, 0.0, 1.0);
                }
            }
            if (!std::isfinite(phase)) {
                phase = (std::atan2(pt.y, pt.x) + kPi) / (2.0 * kPi);
            }
            if (rev_spin) {
                phase = 1.0 - phase;
            }
            const double dt_to_ref = clamp_d((1.0 - phase) * scan_dur, 0.0, max_dt);
            const double xt = pt.x - vx * dt_to_ref;
            const double yt = pt.y - vy * dt_to_ref;
            const double dtheta = wz * dt_to_ref;
            const double c = 1.0 - 0.5 * (dtheta * dtheta);
            const double s = dtheta;
            pt.x = c * xt + s * yt;
            pt.y = -s * xt + c * yt;
        }
    }

    // Post-deskew ROI crop
    thread_local std::vector<double> roi_x;
    thread_local std::vector<double> roi_y;
    thread_local std::vector<double> roi_z;
    thread_local std::vector<double> roi_refl;
    roi_x.clear();
    roi_y.clear();
    roi_z.clear();
    roi_refl.clear();

    const bool has_stripe = (cfg->off_stripe >= 0 && cfg->stripe_datatype > 0);
    for (const auto& pt : cand_pts) {
        if (pt.x >= cfg->roi_x_min && pt.x <= cfg->roi_x_max &&
            pt.y >= cfg->roi_y_min && pt.y <= cfg->roi_y_max &&
            pt.z >= cfg->roi_z_min && pt.z <= cfg->roi_z_max) {
            roi_x.push_back(pt.x);
            roi_y.push_back(pt.y);
            roi_z.push_back(pt.z);
            if (has_stripe) {
                const uint8_t* p = raw_bytes + static_cast<size_t>(pt.orig_idx) * point_step + cfg->off_stripe;
                roi_refl.push_back(read_field_as_double(p, cfg->stripe_datatype));
            }
        }
    }

    const int32_t n_roi = static_cast<int32_t>(roi_x.size());
    if (n_roi == 0) {
        if (out_stats) {
            out_stats->raw_points = raw_count;
            out_stats->kept_points = 0;
            out_stats->self_return_rejects = self_return_rejects;
        }
        return 0;
    }

    int32_t n_kept = cpp_filter_ground_only(
        n_roi,
        roi_x.data(),
        roi_y.data(),
        roi_z.data(),
        has_stripe ? roi_refl.data() : nullptr,
        cfg,
        out_x,
        out_y,
        out_z,
        out_refl,
        max_out
    );

    if (n_kept > cfg->max_points_for_clustering && cfg->max_points_for_clustering > 0) {
        const int32_t stride = static_cast<int32_t>(std::ceil(
            static_cast<double>(n_kept) / static_cast<double>(cfg->max_points_for_clustering)
        ));
        int32_t w = 0;
        for (int32_t r = 0; r < n_kept; r += stride) {
            out_x[w] = out_x[r];
            out_y[w] = out_y[r];
            out_z[w] = out_z[r];
            if (out_refl && has_stripe) {
                out_refl[w] = out_refl[r];
            }
            ++w;
        }
        n_kept = w;
    }

    if (out_stats) {
        out_stats->raw_points = raw_count;
        out_stats->kept_points = n_kept;
        out_stats->self_return_rejects = self_return_rejects;
    }
    return n_kept;
}

FUSION_CPP_EXPORT int32_t cpp_filter_ground_only(
    int32_t n_roi,
    const double* roi_x,
    const double* roi_y,
    const double* roi_z,
    const double* roi_refl,
    const fusion_core_cpp::LidarPipelineConfigC* cfg,
    double* out_x,
    double* out_y,
    double* out_z,
    double* out_refl,
    int32_t max_out
) {
    if (n_roi <= 0 || !roi_x || !roi_y || !roi_z || !cfg || !out_x || !out_y || !out_z || max_out <= 0) {
        return 0;
    }
    const bool has_stripe = (roi_refl != nullptr);

    // Ground filtering (adaptive concentric zone model or flat threshold)
    thread_local std::vector<uint8_t> keep_mask;
    keep_mask.assign(n_roi, 0);

    if (!cfg->enable_adaptive_ground_model) {
        const double gz = cfg->ground_z_max;
        for (int32_t i = 0; i < n_roi; ++i) {
            if (roi_z[i] > gz) keep_mask[i] = 1;
        }
    } else {
        const double step = std::max(0.5, cfg->ground_zone_radial_step_m);
        const double inv_step = 1.0 / step;
        const double max_slope = std::max(0.0, cfg->max_ground_slope_rad);
        const double clearance = std::max(0.01, cfg->ground_clearance_m);
        const double min_spread = std::max(0.45, 1.5 * cfg->cone_max_radius_m);
        const double min_spread_sq = min_spread * min_spread;
        const int32_t bins = std::max(4, cfg->ground_zone_azimuth_bins);
        const double g_z_max = cfg->ground_z_max;
        const double c_pi4 = 0.25 * kPi;
        const double c_3pi4 = 0.75 * kPi;
        const double bin_scale = static_cast<double>(bins) / (2.0 * kPi);

        thread_local std::vector<double> r_xy;
        thread_local std::vector<int32_t> zone_id;
        thread_local std::vector<int32_t> cand_idx;
        r_xy.resize(n_roi);
        zone_id.resize(n_roi);
        cand_idx.clear();

        int32_t max_zone_id = 0;
        for (int32_t i = 0; i < n_roi; ++i) {
            const double x = roi_x[i];
            const double y = roi_y[i];
            const double z = roi_z[i];
            const double r = std::sqrt(x * x + y * y);
            r_xy[i] = r;
            const int32_t ring_idx = static_cast<int32_t>(r * inv_step);
            const double abs_y = std::abs(y) + 1e-10;
            const double abs_x = std::abs(x);
            const double r_ratio = (abs_x - abs_y) / (abs_x + abs_y);
            const double angle = (x >= 0.0) ? (c_pi4 - c_pi4 * r_ratio) : (c_3pi4 + c_pi4 * r_ratio);
            const double az = (y >= 0.0 ? angle : -angle) + kPi;
            int32_t sec_idx = static_cast<int32_t>(az * bin_scale);
            if (sec_idx < 0) sec_idx = 0;
            if (sec_idx >= bins) sec_idx = bins - 1;
            const int32_t zid = ring_idx * bins + sec_idx;
            zone_id[i] = zid;
            if (zid >= max_zone_id) max_zone_id = zid + 1;
            if (z <= (g_z_max + r * max_slope + 0.15)) {
                cand_idx.push_back(i);
            }
        }

        thread_local std::vector<double> zone_z_base;
        thread_local std::vector<double> zone_r_ref;
        thread_local std::vector<double> zone_slope;
        thread_local std::vector<uint8_t> zone_fitted;
        zone_z_base.assign(max_zone_id, g_z_max);
        zone_r_ref.assign(max_zone_id, 0.0);
        zone_slope.assign(max_zone_id, 0.0);
        zone_fitted.assign(max_zone_id, 0);

        if (!cand_idx.empty()) {
            if (cand_idx.size() > 2048) {
                const size_t c_stride = static_cast<size_t>(std::ceil(static_cast<double>(cand_idx.size()) / 2048.0));
                size_t w = 0;
                for (size_t j = 0; j < cand_idx.size(); j += c_stride) {
                    cand_idx[w++] = cand_idx[j];
                }
                cand_idx.resize(w);
            }
            const int32_t min_zone_pts = (cand_idx.size() >= 1024) ? 8 : 4;
            thread_local std::vector<ZoneCandPt> zpts;
            zpts.resize(cand_idx.size());
            for (size_t j = 0; j < cand_idx.size(); ++j) {
                const int32_t idx = cand_idx[j];
                const double z = roi_z[idx];
                int64_t zq = static_cast<int64_t>((z + 50.0) * 1e6);
                if (zq < 0) zq = 0;
                if (zq > 0x7FFFFFFF) zq = 0x7FFFFFFF;
                zpts[j] = {zone_id[idx], static_cast<int32_t>(zq), roi_x[idx], roi_y[idx], z, r_xy[idx]};
            }
            std::sort(zpts.begin(), zpts.end(), [](const ZoneCandPt& a, const ZoneCandPt& b) {
                if (a.zone_id != b.zone_id) return a.zone_id < b.zone_id;
                return a.z_quant < b.z_quant;
            });

            int32_t fitted_count = 0;
            size_t st = 0;
            while (st < zpts.size()) {
                const int32_t uz = zpts[st].zone_id;
                size_t en = st + 1;
                double xmin = zpts[st].x, xmax = zpts[st].x;
                double ymin = zpts[st].y, ymax = zpts[st].y;
                while (en < zpts.size() && zpts[en].zone_id == uz) {
                    const double x = zpts[en].x;
                    const double y = zpts[en].y;
                    if (x < xmin) xmin = x;
                    if (x > xmax) xmax = x;
                    if (y < ymin) ymin = y;
                    if (y > ymax) ymax = y;
                    ++en;
                }
                const int32_t cnt = static_cast<int32_t>(en - st);
                const double dx = xmax - xmin;
                const double dy = ymax - ymin;
                if (cnt >= min_zone_pts && (dx * dx + dy * dy) >= min_spread_sq) {
                    const int32_t p15_off = std::clamp(static_cast<int32_t>(0.15 * cnt), 0, cnt - 1);
                    double z_ground = zpts[st + p15_off].z;
                    const double ring_center_r = ((uz / bins) + 0.5) * step;
                    const double max_shift = ring_center_r * max_slope;
                    z_ground = clamp_d(z_ground, g_z_max - max_shift, g_z_max + max_shift);

                    const int32_t in_cnt = std::clamp(static_cast<int32_t>(0.60 * cnt) + 1, 1, cnt);
                    double sum_r = 0.0, sum_z = 0.0, sum_rr = 0.0, sum_rz = 0.0;
                    for (int32_t k = 0; k < in_cnt; ++k) {
                        const double rk = zpts[st + k].r;
                        const double zk = zpts[st + k].z;
                        sum_r += rk;
                        sum_z += zk;
                        sum_rr += rk * rk;
                        sum_rz += rk * zk;
                    }
                    const double var_r = sum_rr - (sum_r * sum_r) / in_cnt;
                    const double cov_rz = sum_rz - (sum_r * sum_z) / in_cnt;
                    double slope_r = 0.0;
                    if (in_cnt >= 4 && var_r > 0.15) {
                        slope_r = clamp_d(cov_rz / std::max(0.15, var_r), -max_slope, max_slope);
                    }
                    zone_z_base[uz] = z_ground + clearance;
                    zone_r_ref[uz] = ring_center_r;
                    zone_slope[uz] = slope_r;
                    zone_fitted[uz] = 1;
                    ++fitted_count;
                }
                st = en;
            }

            if (fitted_count >= 2) {
                double sum_fr = 0.0, sum_fz = 0.0;
                for (int32_t z = 0; z < max_zone_id; ++z) {
                    if (zone_fitted[z]) {
                        sum_fr += zone_r_ref[z];
                        sum_fz += zone_z_base[z];
                    }
                }
                const double rm = sum_fr / fitted_count;
                const double zm = sum_fz / fitted_count;
                double vr_all = 0.0, cov_all = 0.0;
                for (int32_t z = 0; z < max_zone_id; ++z) {
                    if (zone_fitted[z]) {
                        const double dr = zone_r_ref[z] - rm;
                        vr_all += dr * dr;
                        cov_all += dr * (zone_z_base[z] - zm);
                    }
                }
                if (vr_all > 1.0) {
                    const double global_slope = clamp_d(cov_all / vr_all, -max_slope, max_slope);
                    for (int32_t z = 0; z < max_zone_id; ++z) {
                        if (!zone_fitted[z] && (z / bins) >= 2) {
                            const double far_r = ((z / bins) + 0.5) * step;
                            zone_z_base[z] = zm + global_slope * (far_r - rm);
                            zone_r_ref[z] = far_r;
                            zone_slope[z] = global_slope;
                        }
                    }
                }
            }
        }

        thread_local std::vector<int32_t> skirt_idx;
        thread_local std::vector<int32_t> seed_idx;
        skirt_idx.clear();
        seed_idx.clear();
        const bool do_recovery = (cfg->enable_ground_point_recovery != 0);
        const double far_r_gate = std::max(6.0, cfg->adaptive_range_ref_m - 2.0);
        const double skirt_margin = std::min(0.035, 0.70 * clearance);

        for (int32_t i = 0; i < n_roi; ++i) {
            const int32_t zid = zone_id[i];
            const double thresh = zone_z_base[zid] + zone_slope[zid] * (r_xy[i] - zone_r_ref[zid]);
            const double z = roi_z[i];
            if (z > thresh) {
                keep_mask[i] = 1;
                if (do_recovery && r_xy[i] >= far_r_gate && z <= (thresh + 0.45)) {
                    seed_idx.push_back(i);
                }
            } else if (do_recovery && r_xy[i] >= far_r_gate && z > (thresh - skirt_margin)) {
                skirt_idx.push_back(i);
            }
        }

        if (do_recovery && !skirt_idx.empty() && !seed_idx.empty() && seed_idx.size() <= 256) {
            const double tol2 = cfg->cluster_tolerance_m * cfg->cluster_tolerance_m;
            thread_local std::vector<int32_t> iso_seeds;
            iso_seeds.clear();
            for (size_t a = 0; a < seed_idx.size(); ++a) {
                const double ax = roi_x[seed_idx[a]];
                const double ay = roi_y[seed_idx[a]];
                int32_t nbrs = 0;
                for (size_t b = 0; b < seed_idx.size(); ++b) {
                    const double dx = ax - roi_x[seed_idx[b]];
                    const double dy = ay - roi_y[seed_idx[b]];
                    if (dx * dx + dy * dy <= tol2) {
                        ++nbrs;
                    }
                }
                if (nbrs == 1) {
                    iso_seeds.push_back(seed_idx[a]);
                }
            }
            if (!iso_seeds.empty()) {
                const size_t max_b = std::min<size_t>(256, skirt_idx.size());
                for (size_t b = 0; b < max_b; ++b) {
                    const int32_t b_i = skirt_idx[b];
                    const double bx = roi_x[b_i];
                    const double by = roi_y[b_i];
                    for (int32_t s_i : iso_seeds) {
                        const double dx = bx - roi_x[s_i];
                        const double dy = by - roi_y[s_i];
                        if (dx * dx + dy * dy <= (0.15 * 0.15)) {
                            keep_mask[b_i] = 1;
                            break;
                        }
                    }
                }
            }
        }
    }

    int32_t n_kept = 0;
    for (int32_t i = 0; i < n_roi; ++i) {
        if (keep_mask[i]) {
            if (n_kept < max_out) {
                out_x[n_kept] = roi_x[i];
                out_y[n_kept] = roi_y[i];
                out_z[n_kept] = roi_z[i];
                if (out_refl && has_stripe) {
                    out_refl[n_kept] = roi_refl[i];
                }
                ++n_kept;
            }
        }
    }
    return n_kept;
}

FUSION_CPP_EXPORT int32_t cpp_cluster_and_build(
    int32_t n_pts,
    const double* kx,
    const double* ky,
    const double* kz,
    const double* krefl,
    const fusion_core_cpp::ClusterBuildConfigC* cfg,
    fusion_core_cpp::ClusterDescriptorC* out_clusters,
    int32_t max_clusters
) {
    if (n_pts <= 0 || !kx || !ky || !kz || !cfg || !out_clusters || max_clusters <= 0) {
        return 0;
    }

    const double tol = std::max(0.05, cfg->cluster_tolerance_m);
    const double inv_c = 1.0 / (tol * 0.85);
    double x0 = kx[0], y0 = ky[0];
    for (int32_t i = 1; i < n_pts; ++i) {
        if (kx[i] < x0) x0 = kx[i];
        if (ky[i] < y0) y0 = ky[i];
    }

    thread_local std::vector<int32_t> gx;
    thread_local std::vector<int32_t> gy;
    gx.resize(n_pts);
    gy.resize(n_pts);
    int32_t max_gx = 0, max_gy = 0;
    for (int32_t i = 0; i < n_pts; ++i) {
        const int32_t ix = static_cast<int32_t>((kx[i] - x0) * inv_c);
        const int32_t iy = static_cast<int32_t>((ky[i] - y0) * inv_c);
        gx[i] = ix;
        gy[i] = iy;
        if (ix > max_gx) max_gx = ix;
        if (iy > max_gy) max_gy = iy;
    }

    const int32_t w = max_gx + 2;
    const int32_t h = max_gy + 2;
    thread_local std::vector<int32_t> lbl_grid;
    lbl_grid.assign(static_cast<size_t>(w) * h, 0);
    for (int32_t i = 0; i < n_pts; ++i) {
        lbl_grid[gx[i] * h + gy[i]] = -1;  // occupied unvisited
    }

    int32_t n_comp = 0;
    thread_local std::vector<int32_t> bfs_q;
    for (int32_t r = 0; r < w; ++r) {
        for (int32_t c = 0; c < h; ++c) {
            const int32_t cell_idx = r * h + c;
            if (lbl_grid[cell_idx] != -1) continue;
            ++n_comp;
            lbl_grid[cell_idx] = n_comp;
            bfs_q.clear();
            bfs_q.push_back(cell_idx);
            size_t head = 0;
            while (head < bfs_q.size()) {
                const int32_t cur = bfs_q[head++];
                const int32_t cr = cur / h;
                const int32_t cc = cur % h;
                for (int32_t dr = -1; dr <= 1; ++dr) {
                    const int32_t nr = cr + dr;
                    if (nr < 0 || nr >= w) continue;
                    for (int32_t dc = -1; dc <= 1; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        const int32_t nc = cc + dc;
                        if (nc < 0 || nc >= h) continue;
                        const int32_t nidx = nr * h + nc;
                        if (lbl_grid[nidx] == -1) {
                            lbl_grid[nidx] = n_comp;
                            bfs_q.push_back(nidx);
                        }
                    }
                }
            }
        }
    }

    if (n_comp == 0) return 0;

    thread_local std::vector<int32_t> comp_counts;
    comp_counts.assign(n_comp + 1, 0);
    thread_local std::vector<int32_t> pt_lbl;
    pt_lbl.resize(n_pts);
    for (int32_t i = 0; i < n_pts; ++i) {
        const int32_t l = lbl_grid[gx[i] * h + gy[i]];
        pt_lbl[i] = l;
        ++comp_counts[l];
    }

    const int32_t max_pts = cfg->max_cluster_points;
    const bool do_split = (cfg->enable_cluster_splitting != 0);
    const double split_max_r = (cfg->split_max_radius_m > 0.0) ? cfg->split_max_radius_m : 0.95;
    const double split_min_sep = (cfg->split_min_separation_m > 0.0) ? cfg->split_min_separation_m : 0.28;
    const int32_t comp_max_pts = do_split ? (2 * max_pts) : max_pts;
    const int32_t base_min_pts = cfg->min_cluster_points;
    const bool adaptive = (cfg->enable_adaptive_clustering != 0);
    const int32_t far_min_pts = adaptive ? std::max(1, cfg->min_cluster_points_far) : base_min_pts;
    const double ref_r = cfg->adaptive_range_ref_m;
    const double near_gate_r2 = std::max(0.0, ref_r - 0.5) * std::max(0.0, ref_r - 0.5);

    thread_local std::vector<int32_t> comp_head;
    thread_local std::vector<int32_t> next_pt;
    comp_head.assign(n_comp + 1, -1);
    next_pt.assign(n_pts, -1);
    for (int32_t i = n_pts - 1; i >= 0; --i) {
        const int32_t l = pt_lbl[i];
        const int32_t cnt = comp_counts[l];
        const bool keep = ((cnt >= base_min_pts) ||
                           (adaptive && cnt >= far_min_pts && (kx[i] * kx[i] + ky[i] * ky[i]) >= near_gate_r2)) &&
                          (cnt <= comp_max_pts);
        if (keep) {
            next_pt[i] = comp_head[l];
            comp_head[l] = i;
        }
    }

    thread_local std::vector<fusion_core_cpp::ClusterDescriptorC> valid_clusters;
    valid_clusters.clear();

    const bool has_refl = (cfg->has_refl != 0 && krefl != nullptr && cfg->enable_stripe_classification != 0);
    const bool use_off = (cfg->enable_surface_axis_offset != 0);
    const double axis_off = cfg->surface_axis_offset_m;
    const double c_min_h = cfg->cone_min_height_m;
    const double c_max_h = cfg->cone_max_height_m;
    const double c_max_r = cfg->cone_max_radius_m;

    auto try_emit_cone = [&](const std::vector<int32_t>& idxs) -> bool {
        const int32_t n = static_cast<int32_t>(idxs.size());
        if (n <= 0 || n > max_pts) return false;
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        double zmin = std::numeric_limits<double>::infinity();
        double zmax = -std::numeric_limits<double>::infinity();
        for (int32_t cur : idxs) {
            const double x = kx[cur];
            const double y = ky[cur];
            const double z = kz[cur];
            sum_x += x; sum_y += y; sum_z += z;
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;
        }
        const double xm = sum_x / n;
        const double ym = sum_y / n;
        const double zm = sum_z / n;
        const double d_xy = std::hypot(xm, ym);
        const bool is_far = adaptive && (d_xy >= ref_r);
        const int32_t eff_min_pts = is_far ? far_min_pts : base_min_pts;
        if (n < eff_min_pts) return false;

        const double h_val = zmax - zmin;
        double max_r2 = 0.0;
        double sum_dx2 = 0.0, sum_dy2 = 0.0, sum_dz2 = 0.0, sum_dxy = 0.0;
        double mid_sum = 0.0, out_sum = 0.0;
        int32_t mid_cnt = 0, out_cnt = 0;
        const double inv_h = 1.0 / std::max(1e-6, h_val);
        for (int32_t cur : idxs) {
            const double dx = kx[cur] - xm;
            const double dy = ky[cur] - ym;
            const double dz = kz[cur] - zm;
            const double r2 = dx * dx + dy * dy;
            if (r2 > max_r2) max_r2 = r2;
            sum_dx2 += dx * dx;
            sum_dy2 += dy * dy;
            sum_dz2 += dz * dz;
            sum_dxy += dx * dy;
            if (has_refl) {
                const double znorm = (kz[cur] - zmin) * inv_h;
                if (znorm >= 0.25 && znorm <= 0.75) {
                    mid_sum += krefl[cur];
                    ++mid_cnt;
                } else {
                    out_sum += krefl[cur];
                    ++out_cnt;
                }
            }
        }
        const double rad = std::sqrt(max_r2);
        const double eff_min_h = (is_far && n < base_min_pts && rad <= std::min(c_max_r, 0.15)) ? 0.0 : c_min_h;
        if (h_val < eff_min_h || h_val > c_max_h || rad > c_max_r) {
            return false;
        }
        const double size_score = clamp_d(
            static_cast<double>(n - eff_min_pts) / std::max(1.0, static_cast<double>(max_pts - eff_min_pts)),
            0.0, 1.0
        );
        const double compact_score = clamp_d(1.0 - rad / std::max(0.05, c_max_r), 0.0, 1.0);
        const double height_score = (is_far && n < base_min_pts)
                                        ? 0.65
                                        : clamp_d(1.0 - std::abs(h_val - 0.35) / 0.35, 0.0, 1.0);
        const double geom_q = clamp_d(0.45 * size_score + 0.35 * compact_score + 0.20 * height_score, 0.0, 1.0);
        double cx_o = xm;
        double cy_o = ym;
        if (use_off && d_xy > 0.5) {
            cx_o += axis_off * (cx_o / d_xy);
            cy_o += axis_off * (cy_o / d_xy);
        }
        double stripe_ratio = 1.0;
        if (has_refl && n >= 5 && h_val >= 0.08 && mid_cnt >= 1 && out_cnt >= 1 && (mid_sum + out_sum) > 1.0) {
            stripe_ratio = (mid_sum / mid_cnt) / std::max(1.0, out_sum / out_cnt);
        }
        valid_clusters.push_back({
            cx_o,
            cy_o,
            zm,
            std::max(cfg->covariance_floor_xy, sum_dx2 / n),
            sum_dxy / n,
            std::max(cfg->covariance_floor_xy, sum_dy2 / n),
            std::max(cfg->covariance_floor_z, sum_dz2 / n),
            geom_q,
            stripe_ratio,
            n
        });
        return true;
    };

    thread_local std::vector<int32_t> comp_pts;
    thread_local std::vector<int32_t> split_a;
    thread_local std::vector<int32_t> split_b;

    for (int32_t l = 1; l <= n_comp; ++l) {
        int32_t p = comp_head[l];
        if (p < 0) continue;

        int32_t n = 0;
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        double zmin = std::numeric_limits<double>::infinity();
        double zmax = -std::numeric_limits<double>::infinity();
        for (int32_t cur = p; cur >= 0; cur = next_pt[cur]) {
            const double x = kx[cur];
            const double y = ky[cur];
            const double z = kz[cur];
            sum_x += x;
            sum_y += y;
            sum_z += z;
            if (z < zmin) zmin = z;
            if (z > zmax) zmax = z;
            ++n;
        }

        const double xm = sum_x / n;
        const double ym = sum_y / n;
        const double zm = sum_z / n;
        const double d_xy = std::hypot(xm, ym);
        const bool is_far = adaptive && (d_xy >= ref_r);
        const int32_t eff_min_pts = is_far ? far_min_pts : base_min_pts;
        if (n < eff_min_pts) continue;

        const double h_val = zmax - zmin;
        double max_r2 = 0.0;
        double sum_dx2 = 0.0, sum_dy2 = 0.0, sum_dz2 = 0.0, sum_dxy = 0.0;
        double mid_sum = 0.0, out_sum = 0.0;
        int32_t mid_cnt = 0, out_cnt = 0;
        const double inv_h = 1.0 / std::max(1e-6, h_val);

        for (int32_t cur = p; cur >= 0; cur = next_pt[cur]) {
            const double dx = kx[cur] - xm;
            const double dy = ky[cur] - ym;
            const double dz = kz[cur] - zm;
            const double r2 = dx * dx + dy * dy;
            if (r2 > max_r2) max_r2 = r2;
            sum_dx2 += dx * dx;
            sum_dy2 += dy * dy;
            sum_dz2 += dz * dz;
            sum_dxy += dx * dy;
            if (has_refl) {
                const double znorm = (kz[cur] - zmin) * inv_h;
                if (znorm >= 0.25 && znorm <= 0.75) {
                    mid_sum += krefl[cur];
                    ++mid_cnt;
                } else {
                    out_sum += krefl[cur];
                    ++out_cnt;
                }
            }
        }

        const double rad = std::sqrt(max_r2);
        const double eff_min_h = (is_far && n < base_min_pts && rad <= std::min(c_max_r, 0.15)) ? 0.0 : c_min_h;
        if (n <= max_pts && h_val >= eff_min_h && h_val <= c_max_h && rad <= c_max_r) {
            const double size_score = clamp_d(
                static_cast<double>(n - eff_min_pts) / std::max(1.0, static_cast<double>(max_pts - eff_min_pts)),
                0.0, 1.0
            );
            const double compact_score = clamp_d(1.0 - rad / std::max(0.05, c_max_r), 0.0, 1.0);
            const double height_score = (is_far && n < base_min_pts)
                                            ? 0.65
                                            : clamp_d(1.0 - std::abs(h_val - 0.35) / 0.35, 0.0, 1.0);
            const double geom_q = clamp_d(0.45 * size_score + 0.35 * compact_score + 0.20 * height_score, 0.0, 1.0);

            double cx_o = xm;
            double cy_o = ym;
            if (use_off && d_xy > 0.5) {
                cx_o += axis_off * (cx_o / d_xy);
                cy_o += axis_off * (cy_o / d_xy);
            }

            double stripe_ratio = 1.0;
            if (has_refl && n >= 5 && h_val >= 0.08 && mid_cnt >= 1 && out_cnt >= 1 && (mid_sum + out_sum) > 1.0) {
                stripe_ratio = (mid_sum / mid_cnt) / std::max(1.0, out_sum / out_cnt);
            }

            valid_clusters.push_back({
                cx_o,
                cy_o,
                zm,
                std::max(cfg->covariance_floor_xy, sum_dx2 / n),
                sum_dxy / n,
                std::max(cfg->covariance_floor_xy, sum_dy2 / n),
                std::max(cfg->covariance_floor_z, sum_dz2 / n),
                geom_q,
                stripe_ratio,
                n
            });
            continue;
        }

        // Targeted 2-means split only for oversized components that could represent 2 close cones.
        if (do_split && h_val >= eff_min_h && h_val <= c_max_h &&
            rad > c_max_r && rad <= split_max_r && n >= 2 * eff_min_pts) {
            comp_pts.clear();
            int32_t seed_a = p;
            double best_a_r2 = -1.0;
            for (int32_t cur = p; cur >= 0; cur = next_pt[cur]) {
                comp_pts.push_back(cur);
                const double d2 = (kx[cur] - xm) * (kx[cur] - xm) + (ky[cur] - ym) * (ky[cur] - ym);
                if (d2 > best_a_r2) { best_a_r2 = d2; seed_a = cur; }
            }
            int32_t seed_b = seed_a;
            double best_b_r2 = -1.0;
            for (int32_t cur : comp_pts) {
                const double d2 = (kx[cur] - kx[seed_a]) * (kx[cur] - kx[seed_a]) +
                                  (ky[cur] - ky[seed_a]) * (ky[cur] - ky[seed_a]);
                if (d2 > best_b_r2) { best_b_r2 = d2; seed_b = cur; }
            }
            if (best_b_r2 >= split_min_sep * split_min_sep) {
                double ax = kx[seed_a], ay = ky[seed_a];
                double bx = kx[seed_b], by = ky[seed_b];
                for (int it = 0; it < 6; ++it) {
                    double sax = 0.0, say = 0.0, sbx = 0.0, sby = 0.0;
                    int32_t ca = 0, cb = 0;
                    for (int32_t cur : comp_pts) {
                        const double da = (kx[cur] - ax) * (kx[cur] - ax) + (ky[cur] - ay) * (ky[cur] - ay);
                        const double db = (kx[cur] - bx) * (kx[cur] - bx) + (ky[cur] - by) * (ky[cur] - by);
                        if (da <= db) { sax += kx[cur]; say += ky[cur]; ++ca; }
                        else { sbx += kx[cur]; sby += ky[cur]; ++cb; }
                    }
                    if (ca == 0 || cb == 0) break;
                    const double nax = sax / ca, nay = say / ca;
                    const double nbx = sbx / cb, nby = sby / cb;
                    const bool conv = (std::abs(nax - ax) < 1e-4 && std::abs(nay - ay) < 1e-4 &&
                                       std::abs(nbx - bx) < 1e-4 && std::abs(nby - by) < 1e-4);
                    ax = nax; ay = nay; bx = nbx; by = nby;
                    if (conv) break;
                }
                if (std::hypot(ax - bx, ay - by) >= split_min_sep) {
                    split_a.clear();
                    split_b.clear();
                    for (int32_t cur : comp_pts) {
                        const double da = (kx[cur] - ax) * (kx[cur] - ax) + (ky[cur] - ay) * (ky[cur] - ay);
                        const double db = (kx[cur] - bx) * (kx[cur] - bx) + (ky[cur] - by) * (ky[cur] - by);
                        if (da <= db) split_a.push_back(cur);
                        else split_b.push_back(cur);
                    }
                    try_emit_cone(split_a);
                    try_emit_cone(split_b);
                }
            }
        }
    }

    std::stable_sort(valid_clusters.begin(), valid_clusters.end(),
                     [](const fusion_core_cpp::ClusterDescriptorC& a,
                        const fusion_core_cpp::ClusterDescriptorC& b) {
                         return a.point_count > b.point_count;
                     });

    const int32_t out_n = std::min(static_cast<int32_t>(valid_clusters.size()), max_clusters);
    for (int32_t i = 0; i < out_n; ++i) {
        out_clusters[i] = valid_clusters[i];
    }
    return out_n;
}

namespace {

inline void normalize_probs4(const double* in, double* out) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        out[i] = (in && std::isfinite(in[i]) && in[i] > 0.0) ? in[i] : 0.0;
        s += out[i];
    }
    if (s <= 1e-9) {
        out[0] = out[1] = out[2] = out[3] = 0.25;
        return;
    }
    if (std::abs(s - 1.0) <= 1e-6) {
        return;
    }
    const double inv = 1.0 / s;
    for (int i = 0; i < 4; ++i) {
        out[i] *= inv;
    }
}

inline double normalized_information4(const double* probs) {
    double p[4];
    normalize_probs4(probs, p);
    double entropy = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (p[i] > 1e-12) {
            entropy -= p[i] * std::log(p[i]);
        }
    }
    constexpr double max_entropy = 1.3862943611198906;  // ln(4.0)
    return clamp_d(1.0 - (entropy / max_entropy), 0.0, 1.0);
}

// Kuhn-Munkres / Jonker-Volgenant O(n^2 m) minimum cost bipartite matching.
// Returns matched column index for each row (-1 if unmatched).
std::vector<int32_t> solve_hungarian_min_cost(
    int32_t n_rows,
    int32_t n_cols,
    const std::vector<double>& cost_flat
) {
    std::vector<int32_t> row_to_col(std::max(0, n_rows), -1);
    if (n_rows <= 0 || n_cols <= 0) {
        return row_to_col;
    }
    const bool transposed = (n_rows > n_cols);
    const int32_t R = transposed ? n_cols : n_rows;
    const int32_t C = transposed ? n_rows : n_cols;

    auto get_cost = [&](int32_t r, int32_t c) -> double {
        return transposed ? cost_flat[c * n_cols + r] : cost_flat[r * n_cols + c];
    };

    const double kInf = 1e18;
    std::vector<double> u(R + 1, 0.0);
    std::vector<double> v(C + 1, 0.0);
    std::vector<int32_t> p(C + 1, 0);
    std::vector<int32_t> way(C + 1, 0);

    for (int32_t i = 1; i <= R; ++i) {
        p[0] = i;
        int32_t j0 = 0;
        std::vector<double> minv(C + 1, kInf);
        std::vector<uint8_t> used(C + 1, 0);
        do {
            used[j0] = 1;
            const int32_t i0 = p[j0];
            double delta = kInf;
            int32_t j1 = 0;
            for (int32_t j = 1; j <= C; ++j) {
                if (!used[j]) {
                    const double cur = get_cost(i0 - 1, j - 1) - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int32_t j = 0; j <= C; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int32_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    for (int32_t j = 1; j <= C; ++j) {
        if (p[j] > 0) {
            if (transposed) {
                row_to_col[j - 1] = p[j] - 1;
            } else {
                row_to_col[p[j] - 1] = j - 1;
            }
        }
    }
    return row_to_col;
}

inline bool project_lidar_point_c(
    double px,
    double py,
    double pz,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    double& out_u,
    double& out_v,
    double& out_depth
) {
    const double* R = ext->r;
    const double* t = ext->t;
    const double x_cam = R[0] * px + R[1] * py + R[2] * pz + t[0];
    const double y_cam = R[3] * px + R[4] * py + R[5] * pz + t[1];
    const double z_cam = R[6] * px + R[7] * py + R[8] * pz + t[2];
    if (z_cam <= ext->min_depth_m || z_cam >= ext->max_depth_m) {
        return false;
    }
    const double xn = x_cam / z_cam;
    const double yn = y_cam / z_cam;
    double xd = xn;
    double yd = yn;
    if (intr->distortion_model == 1) {
        const double r = std::hypot(xn, yn);
        if (r > 1e-12) {
            const double theta = std::atan(r);
            const double t2 = theta * theta;
            const double t4 = t2 * t2;
            const double t6 = t4 * t2;
            const double t8 = t4 * t4;
            const double theta_d = theta * (1.0 + intr->k1 * t2 + intr->k2 * t4 + intr->k3 * t6 + intr->k4 * t8);
            const double scale = theta_d / r;
            xd = xn * scale;
            yd = yn * scale;
        }
    } else if (std::abs(intr->k1) > 1e-12 || std::abs(intr->k2) > 1e-12 ||
               std::abs(intr->p1) > 1e-12 || std::abs(intr->p2) > 1e-12 ||
               std::abs(intr->k3) > 1e-12) {
        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = 1.0 + intr->k1 * r2 + intr->k2 * r4 + intr->k3 * r6;
        const double xy2 = 2.0 * xn * yn;
        xd = xn * radial + intr->p1 * xy2 + intr->p2 * (r2 + 2.0 * xn * xn);
        yd = yn * radial + intr->p1 * (r2 + 2.0 * yn * yn) + intr->p2 * xy2;
    }
    const double u = intr->fx * xd + intr->cx;
    const double v = intr->fy * yd + intr->cy;
    if (u < 0.0 || u >= static_cast<double>(intr->width) ||
        v < 0.0 || v >= static_cast<double>(intr->height)) {
        return false;
    }
    out_u = u;
    out_v = v;
    out_depth = z_cam;
    return true;
}

inline bool point_in_polygon_c(int32_t n, const double* px, const double* py, double u, double v) {
    bool inside = false;
    for (int32_t i = 0, j = n - 1; i < n; j = i++) {
        const double yi = py[i];
        const double yj = py[j];
        const bool intersect = ((yi > v) != (yj > v)) &&
                               (u < (px[j] - px[i]) * (v - yi) / ((yj - yi) + 1e-12) + px[i]);
        if (intersect) inside = !inside;
    }
    return inside;
}

inline double min_dist_to_polygon_edges_c(int32_t n, const double* px, const double* py, double u, double v) {
    double best_d2 = 1e18;
    for (int32_t i = 0, j = n - 1; i < n; j = i++) {
        const double x1 = px[j], y1 = py[j];
        const double x2 = px[i], y2 = py[i];
        const double dx = x2 - x1;
        const double dy = y2 - y1;
        const double len2 = dx * dx + dy * dy;
        double t = 0.0;
        if (len2 > 1e-12) {
            t = clamp_d(((u - x1) * dx + (v - y1) * dy) / len2, 0.0, 1.0);
        }
        const double proj_x = x1 + t * dx;
        const double proj_y = y1 + t * dy;
        const double d2 = (u - proj_x) * (u - proj_x) + (v - proj_y) * (v - proj_y);
        if (d2 < best_d2) best_d2 = d2;
    }
    return std::sqrt(best_d2);
}

inline double detection_likelihood_c(
    const fusion_core_cpp::Detection2DC& det,
    const fusion_core_cpp::ExtrinsicsC* ext,
    double u,
    double v
) {
    const double width = std::max(1.0, det.xmax - det.xmin);
    const double height = std::max(1.0, det.ymax - det.ymin);
    const double cx = 0.5 * (det.xmin + det.xmax);
    const double cy = 0.5 * (det.ymin + det.ymax);
    const double half_w = 0.5 * width;
    const double half_h = 0.5 * height;

    // 1. Explicit polygon / keypoint mask mode (e.g. 3-keypoint triangle or seg contour)
    if (det.polygon_count >= 3 && det.polygon_count <= 16) {
        const int32_t n = det.polygon_count;
        const bool inside = point_in_polygon_c(n, det.polygon_u, det.polygon_v, u, v);
        if (ext->soft_likelihood_enabled == 0) {
            return inside ? 1.0 : 0.0;
        }
        double p_cu = 0.0, p_cv = 0.0;
        double u_min = det.polygon_u[0], u_max = det.polygon_u[0];
        double v_min = det.polygon_v[0], v_max = det.polygon_v[0];
        for (int32_t i = 0; i < n; ++i) {
            p_cu += det.polygon_u[i];
            p_cv += det.polygon_v[i];
            if (det.polygon_u[i] < u_min) u_min = det.polygon_u[i];
            if (det.polygon_u[i] > u_max) u_max = det.polygon_u[i];
            if (det.polygon_v[i] < v_min) v_min = det.polygon_v[i];
            if (det.polygon_v[i] > v_max) v_max = det.polygon_v[i];
        }
        p_cu /= static_cast<double>(n);
        p_cv /= static_cast<double>(n);
        const double pw = std::max(1.0, u_max - u_min);
        const double ph = std::max(1.0, v_max - v_min);
        if (!inside) {
            const double d_edge = min_dist_to_polygon_edges_c(n, det.polygon_u, det.polygon_v, u, v);
            const double max_outside = std::max(2.0, std::max(0.0, ext->soft_likelihood_outside_ratio - 1.0) * 0.5 * std::min(pw, ph));
            if (d_edge > max_outside) {
                return 0.0;
            }
        }
        const double sigma_u = std::max(1.0, ext->soft_likelihood_sigma_scale * pw);
        const double sigma_v = std::max(1.0, ext->soft_likelihood_sigma_scale * ph);
        const double du = (u - p_cu) / sigma_u;
        const double dv = (v - p_cv) / sigma_v;
        double val = std::exp(-0.5 * (du * du + dv * dv));
        if (!inside) val *= 0.65;
        if (val < ext->soft_likelihood_min_value) return 0.0;
        return clamp_d(val, 0.0, 1.0);
    }

    // 2. Canonical upright triangular cone silhouette inside bounding box
    if (ext->enable_triangular_cone_mask != 0) {
        const double apex_ratio = clamp_d(ext->cone_mask_apex_width_ratio, 0.0, 1.0);
        const double t_vert = clamp_d((v - det.ymin) / height, 0.0, 1.0);
        const double half_w_t = std::max(0.5, half_w * (apex_ratio + (1.0 - apex_ratio) * t_vert));

        if (ext->soft_likelihood_enabled == 0) {
            if (v < det.ymin || v > det.ymax || std::abs(u - cx) > half_w_t) {
                return 0.0;
            }
            return 1.0;
        }

        const double outside_w = std::max(1.0, ext->soft_likelihood_outside_ratio * half_w_t);
        const double outside_h = std::max(1.0, ext->soft_likelihood_outside_ratio * half_h);
        if (std::abs(u - cx) > outside_w || std::abs(v - cy) > outside_h) {
            return 0.0;
        }
        const double sigma_u = std::max(1.0, ext->soft_likelihood_sigma_scale * (2.0 * half_w_t));
        const double sigma_v = std::max(1.0, ext->soft_likelihood_sigma_scale * height);
        const double du = (u - cx) / sigma_u;
        const double dv = (v - cy) / sigma_v;
        const double val = std::exp(-0.5 * (du * du + dv * dv));
        if (val < ext->soft_likelihood_min_value) {
            return 0.0;
        }
        return clamp_d(val, 0.0, 1.0);
    }

    // 3. Standard rectangular bounding box likelihood
    if (ext->soft_likelihood_enabled == 0) {
        if (u < det.xmin || u > det.xmax || v < det.ymin || v > det.ymax) {
            return 0.0;
        }
        return 1.0;
    }

    const double outside_w = std::max(1.0, ext->soft_likelihood_outside_ratio * half_w);
    const double outside_h = std::max(1.0, ext->soft_likelihood_outside_ratio * half_h);
    if (std::abs(u - cx) > outside_w || std::abs(v - cy) > outside_h) {
        return 0.0;
    }
    const double sigma_u = std::max(1.0, ext->soft_likelihood_sigma_scale * width);
    const double sigma_v = std::max(1.0, ext->soft_likelihood_sigma_scale * height);
    const double du = (u - cx) / sigma_u;
    const double dv = (v - cy) / sigma_v;
    const double val = std::exp(-0.5 * (du * du + dv * dv));
    if (val < ext->soft_likelihood_min_value) {
        return 0.0;
    }
    return clamp_d(val, 0.0, 1.0);
}

struct ProjectedSampleC {
    double u;
    double v;
    double depth;
};

inline void sample_cluster_points_c(
    const fusion_core_cpp::ClusterInputC& c,
    const fusion_core_cpp::ExtrinsicsC* ext,
    ProjectedSampleC* pts3d,
    int32_t& out_count
) {
    double cxx = c.cov_xx;
    double cyy = c.cov_yy;
    double czz = c.cov_zz;
    if (std::abs(cxx) < 1e-12 && std::abs(cyy) < 1e-12 && std::abs(czz) < 1e-12) {
        cxx = 0.05;
        cyy = 0.05;
        czz = 0.1;
    }
    const double var_x = std::max(1e-4, cxx);
    const double var_y = std::max(1e-4, cyy);
    const double var_z = std::max(1e-4, czz);
    const double radius = ext->sample_radius_scale * std::max(0.05, std::sqrt(std::max(var_x, var_y)));
    const double z_off = 0.5 * std::sqrt(var_z);
    const double x = c.cx, y = c.cy, z = c.cz;
    pts3d[0] = {x, y, z};
    pts3d[1] = {x + radius, y, z};
    pts3d[2] = {x - radius, y, z};
    pts3d[3] = {x, y + radius, z};
    pts3d[4] = {x, y - radius, z};
    pts3d[5] = {x, y, z + z_off};
    pts3d[6] = {x, y, z - z_off};
    out_count = 7;
    if (ext->enable_frustum_hull_sampling != 0) {
        constexpr double cone_r = 0.095;
        pts3d[7] = {x, y, z + 0.16};
        pts3d[8] = {x, y + cone_r, z - 0.06};
        pts3d[9] = {x, y - cone_r, z - 0.06};
        pts3d[10] = {x, y + 0.5 * cone_r, z + 0.08};
        pts3d[11] = {x, y - 0.5 * cone_r, z + 0.08};
        out_count = 12;
    }
}

}  // namespace

FUSION_CPP_EXPORT int32_t cpp_estimate_batch_semantics(
    int32_t n_clusters,
    const fusion_core_cpp::ClusterInputC* clusters,
    int32_t n_dets,
    const fusion_core_cpp::Detection2DC* dets,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    const double* fallback_probs4,
    fusion_core_cpp::ClusterSemanticC* out_semantics
) {
    if (!out_semantics || !intr || !ext || n_clusters <= 0 || !clusters) {
        return 0;
    }
    double fallback[4];
    normalize_probs4(fallback_probs4, fallback);

    if (n_dets <= 0 || !dets) {
        for (int32_t c_idx = 0; c_idx < n_clusters; ++c_idx) {
            fusion_core_cpp::ClusterSemanticC& o = out_semantics[c_idx];
            for (int k = 0; k < 4; ++k) o.class_probs[k] = fallback[k];
            o.semantic_weight = clamp_d(ext->no_match_semantic_weight, 0.0, 1.0);
            o.projected_samples = 0;
            o.matched_samples = 0;
            o.matched_weight_sum = 0.0;
            o.best_confidence = 0.0;
            o.residual_samples = 0;
            o.residual_sum_px = 0.0;
            o.residual_max_px = 0.0;
            o.matched_detection_idx = -1;
        }
        return n_clusters;
    }

    std::vector<std::vector<ProjectedSampleC>> all_proj(static_cast<size_t>(n_clusters));
    for (int32_t c_idx = 0; c_idx < n_clusters; ++c_idx) {
        ProjectedSampleC samples3d[12];
        int32_t n_samp = 0;
        sample_cluster_points_c(clusters[c_idx], ext, samples3d, n_samp);
        auto& proj_vec = all_proj[static_cast<size_t>(c_idx)];
        proj_vec.reserve(static_cast<size_t>(n_samp));
        for (int32_t s = 0; s < n_samp; ++s) {
            double u = 0.0, v = 0.0, depth = 0.0;
            if (project_lidar_point_c(samples3d[s].u, samples3d[s].v, samples3d[s].depth,
                                      intr, ext, u, v, depth)) {
                proj_vec.push_back({u, v, depth});
            }
        }
    }

    constexpr double kDisallowedCost = 1e6;
    std::vector<double> cost_mat(static_cast<size_t>(n_clusters) * static_cast<size_t>(n_dets),
                                 kDisallowedCost);
    std::vector<uint8_t> valid_pair(static_cast<size_t>(n_clusters) * static_cast<size_t>(n_dets), 0);
    bool any_valid = false;

    for (int32_t c_idx = 0; c_idx < n_clusters; ++c_idx) {
        const auto& proj_pts = all_proj[static_cast<size_t>(c_idx)];
        if (proj_pts.empty()) continue;
        for (int32_t d_idx = 0; d_idx < n_dets; ++d_idx) {
            const auto& det = dets[d_idx];
            const double width = det.xmax - det.xmin;
            const double height = det.ymax - det.ymin;
            const double area = std::max(1.0, width * height);
            const double inv_sqrt_area = 1.0 / std::sqrt(area);
            const double center_u = 0.5 * (det.xmin + det.xmax);
            const double center_v = 0.5 * (det.ymin + det.ymax);
            const double diag_px = std::max(4.0, std::hypot(width, height));

            double w_sum = 0.0;
            double res_norm_sum = 0.0;
            int32_t matched_cnt = 0;
            for (const auto& pr : proj_pts) {
                const double lk = detection_likelihood_c(det, ext, pr.u, pr.v);
                if (lk <= 0.0) continue;
                const double dw = clamp_d(1.0 - pr.depth / std::max(1.0, ext->max_depth_m), 0.2, 1.0);
                w_sum += det.confidence * lk * inv_sqrt_area * dw;
                res_norm_sum += std::hypot(pr.u - center_u, pr.v - center_v) / diag_px;
                ++matched_cnt;
            }
            if (matched_cnt > 0 && w_sum > 1e-9) {
                const double mean_res_norm = res_norm_sum / static_cast<double>(matched_cnt);
                const double cost = std::max(0.01, 10.0 / (1.0 + 25.0 * w_sum) + 0.5 * mean_res_norm);
                const size_t flat_idx = static_cast<size_t>(c_idx) * static_cast<size_t>(n_dets) +
                                        static_cast<size_t>(d_idx);
                cost_mat[flat_idx] = cost;
                valid_pair[flat_idx] = 1;
                any_valid = true;
            }
        }
    }

    std::vector<int32_t> assigned_det(static_cast<size_t>(n_clusters), -1);
    if (any_valid) {
        auto match = solve_hungarian_min_cost(n_clusters, n_dets, cost_mat);
        for (int32_t c_idx = 0; c_idx < n_clusters; ++c_idx) {
            const int32_t d_idx = match[static_cast<size_t>(c_idx)];
            if (d_idx >= 0 && d_idx < n_dets) {
                const size_t flat_idx = static_cast<size_t>(c_idx) * static_cast<size_t>(n_dets) +
                                        static_cast<size_t>(d_idx);
                if (valid_pair[flat_idx] != 0) {
                    assigned_det[static_cast<size_t>(c_idx)] = d_idx;
                }
            }
        }
    }

    for (int32_t c_idx = 0; c_idx < n_clusters; ++c_idx) {
        const auto& proj_pts = all_proj[static_cast<size_t>(c_idx)];
        const int32_t target_d = assigned_det[static_cast<size_t>(c_idx)];
        const int32_t projected_samples = static_cast<int32_t>(proj_pts.size());

        int32_t matched_samples = 0;
        double matched_weight_sum = 0.0;
        double best_confidence = 0.0;
        int32_t residual_samples = 0;
        double residual_sum_px = 0.0;
        double residual_max_px = 0.0;
        double acc[4] = {0.0, 0.0, 0.0, 0.0};

        if (target_d >= 0) {
            const auto& det = dets[target_d];
            double norm_det_p[4];
            normalize_probs4(det.class_probs, norm_det_p);
            const double area = std::max(1.0, (det.xmax - det.xmin) * (det.ymax - det.ymin));
            const double inv_sqrt_area = 1.0 / std::sqrt(area);
            const double center_u = 0.5 * (det.xmin + det.xmax);
            const double center_v = 0.5 * (det.ymin + det.ymax);

            for (const auto& pr : proj_pts) {
                const double lk = detection_likelihood_c(det, ext, pr.u, pr.v);
                if (lk <= 0.0) continue;
                const double det_weight = det.confidence * lk * inv_sqrt_area;
                if (det_weight <= 1e-9) continue;

                ++matched_samples;
                const double dw = clamp_d(1.0 - pr.depth / std::max(1.0, ext->max_depth_m), 0.2, 1.0);
                const double weight = det_weight * dw;
                matched_weight_sum += weight;
                best_confidence = std::max(best_confidence, det.confidence);

                const double res_px = std::hypot(pr.u - center_u, pr.v - center_v);
                ++residual_samples;
                residual_sum_px += res_px;
                residual_max_px = std::max(residual_max_px, res_px);

                for (int k = 0; k < 4; ++k) {
                    acc[k] += weight * norm_det_p[k];
                }
            }
        }

        fusion_core_cpp::ClusterSemanticC& o = out_semantics[c_idx];
        o.projected_samples = projected_samples;
        o.matched_samples = matched_samples;
        o.matched_weight_sum = matched_weight_sum;
        o.best_confidence = best_confidence;
        o.residual_samples = residual_samples;
        o.residual_sum_px = residual_sum_px;
        o.residual_max_px = residual_max_px;

        if (matched_weight_sum <= 1e-9) {
            for (int k = 0; k < 4; ++k) o.class_probs[k] = fallback[k];
            o.semantic_weight = clamp_d(ext->no_match_semantic_weight, 0.0, 1.0);
            o.matched_detection_idx = -1;
        } else {
            double raw_p[4];
            for (int k = 0; k < 4; ++k) raw_p[k] = acc[k] / matched_weight_sum;
            normalize_probs4(raw_p, o.class_probs);
            const double coverage = (projected_samples > 0)
                                        ? static_cast<double>(matched_samples) /
                                              std::max(1.0, static_cast<double>(projected_samples))
                                        : 0.0;
            o.semantic_weight = clamp_d(
                ext->min_semantic_weight + (1.0 - ext->min_semantic_weight) * coverage * best_confidence,
                ext->min_semantic_weight,
                1.0
            );
            o.matched_detection_idx = target_d;
        }
    }

    return n_clusters;
}

namespace {

inline void undistort_normalized_c(
    double xd,
    double yd,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    double& out_xn,
    double& out_yn
) {
    if (intr->distortion_model == 1) {
        const double theta_d = std::hypot(xd, yd);
        if (theta_d <= 1e-12) {
            out_xn = xd;
            out_yn = yd;
            return;
        }
        double theta = theta_d;
        for (int it = 0; it < 6; ++it) {
            const double t2 = theta * theta;
            const double t4 = t2 * t2;
            const double t6 = t4 * t2;
            const double t8 = t4 * t4;
            const double f = theta * (1.0 + intr->k1 * t2 + intr->k2 * t4 + intr->k3 * t6 + intr->k4 * t8) - theta_d;
            const double df = 1.0 + 3.0 * intr->k1 * t2 + 5.0 * intr->k2 * t4 + 7.0 * intr->k3 * t6 + 9.0 * intr->k4 * t8;
            if (std::abs(df) <= 1e-9) break;
            theta -= f / df;
        }
        theta = std::max(0.0, std::min(1.55, theta));
        const double scale = std::tan(theta) / theta_d;
        out_xn = xd * scale;
        out_yn = yd * scale;
        return;
    }
    if (std::abs(intr->k1) <= 1e-12 && std::abs(intr->k2) <= 1e-12 &&
        std::abs(intr->p1) <= 1e-12 && std::abs(intr->p2) <= 1e-12 &&
        std::abs(intr->k3) <= 1e-12) {
        out_xn = xd;
        out_yn = yd;
        return;
    }
    double xn = xd;
    double yn = yd;
    for (int it = 0; it < 5; ++it) {
        const double r2 = xn * xn + yn * yn;
        const double r4 = r2 * r2;
        const double r6 = r4 * r2;
        const double radial = std::max(1e-4, 1.0 + intr->k1 * r2 + intr->k2 * r4 + intr->k3 * r6);
        const double xy2 = 2.0 * xn * yn;
        const double dx_tang = intr->p1 * xy2 + intr->p2 * (r2 + 2.0 * xn * xn);
        const double dy_tang = intr->p1 * (r2 + 2.0 * yn * yn) + intr->p2 * xy2;
        xn = (xd - dx_tang) / radial;
        yn = (yd - dy_tang) / radial;
    }
    out_xn = xn;
    out_yn = yn;
}

inline bool sample_stereo_depth_for_det_c(
    const fusion_core_cpp::Detection2DC& det,
    const fusion_core_cpp::StereoDepthImageC* sd,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    double& out_depth_m,
    double& out_sigma_m
) {
    if (!sd || !sd->data || sd->encoding == 0 || sd->width <= 0 || sd->height <= 0 || sd->row_step_bytes <= 0) {
        return false;
    }
    const double w = std::max(1.0, det.xmax - det.xmin);
    const double h = std::max(1.0, det.ymax - det.ymin);
    const double cx = 0.5 * (det.xmin + det.xmax);
    const double v0 = std::max(0.0, det.ymin + 0.25 * h);
    const double v1 = std::min(static_cast<double>(std::max(1, intr->height) - 1), det.ymin + 0.85 * h);
    if (v1 <= v0) {
        return false;
    }

    thread_local std::vector<double> samples;
    samples.clear();
    samples.reserve(49);

    const double min_d = std::max(0.1, sd->min_depth_m);
    const double max_d = std::max(min_d + 0.1, sd->max_depth_m);
    const double baseline = (sd->baseline_m > 1e-4) ? sd->baseline_m : 0.12;
    const double disp_sigma = (sd->disparity_sigma_px > 1e-4) ? sd->disparity_sigma_px : 0.5;
    const double fx = std::max(1.0, intr->fx);
    const double scale_x = static_cast<double>(sd->width) / static_cast<double>(std::max(1, intr->width));
    const double scale_y = static_cast<double>(sd->height) / static_cast<double>(std::max(1, intr->height));

    constexpr int kGrid = 7;
    for (int gy = 0; gy < kGrid; ++gy) {
        const double fy_t = (static_cast<double>(gy) + 0.5) / static_cast<double>(kGrid);
        const double v_img = v0 + fy_t * (v1 - v0);
        const int32_t py = std::clamp(static_cast<int32_t>(std::lround(v_img * scale_y)), 0, sd->height - 1);

        // Taper horizontal sampling window to stay strictly inside the cone body
        const double t_cone = clamp_d((v_img - det.ymin) / h, 0.0, 1.0);
        const double half_span = std::max(1.0, 0.30 * w * (0.20 + 0.80 * t_cone));
        const double u_lo = std::max(0.0, cx - half_span);
        const double u_hi = std::min(static_cast<double>(intr->width - 1), cx + half_span);

        for (int gx = 0; gx < kGrid; ++gx) {
            const double fx_t = (static_cast<double>(gx) + 0.5) / static_cast<double>(kGrid);
            const double u_img = u_lo + fx_t * (u_hi - u_lo);
            if (det.polygon_count >= 3 && det.polygon_count <= 16) {
                if (!point_in_polygon_c(det.polygon_count, det.polygon_u, det.polygon_v, u_img, v_img)) {
                    continue;
                }
            }
            const int32_t px = std::clamp(static_cast<int32_t>(std::lround(u_img * scale_x)), 0, sd->width - 1);
            const uint8_t* row_ptr = sd->data + static_cast<size_t>(py) * static_cast<size_t>(sd->row_step_bytes);
            double z_val = std::numeric_limits<double>::quiet_NaN();
            if (sd->encoding == 1) {  // 32FC1 meters
                float raw_f = 0.0f;
                std::memcpy(&raw_f, row_ptr + static_cast<size_t>(px) * 4, 4);
                z_val = static_cast<double>(raw_f) * ((sd->depth_scale > 0.0) ? sd->depth_scale : 1.0);
            } else if (sd->encoding == 2) {  // 16UC1 millimeters
                uint16_t raw_u = 0;
                std::memcpy(&raw_u, row_ptr + static_cast<size_t>(px) * 2, 2);
                if (raw_u > 0) {
                    z_val = static_cast<double>(raw_u) * ((sd->depth_scale > 0.0) ? sd->depth_scale : 0.001);
                }
            } else if (sd->encoding == 3) {  // 32FC1 disparity (px)
                float disp = 0.0f;
                std::memcpy(&disp, row_ptr + static_cast<size_t>(px) * 4, 4);
                if (std::isfinite(disp) && disp > 0.25f) {
                    z_val = (fx * baseline) / static_cast<double>(disp);
                }
            }
            if (std::isfinite(z_val) && z_val >= min_d && z_val <= max_d) {
                samples.push_back(z_val);
            }
        }
    }

    if (samples.size() < 3) {
        return false;
    }
    std::sort(samples.begin(), samples.end());
    const size_t n = samples.size();
    // 35th percentile foreground-biased cone body depth (rejects far asphalt bleed)
    const size_t idx_p20 = static_cast<size_t>(0.20 * static_cast<double>(n - 1));
    const size_t idx_p35 = static_cast<size_t>(0.35 * static_cast<double>(n - 1));
    const size_t idx_p55 = static_cast<size_t>(0.55 * static_cast<double>(n - 1));
    double sum_core = 0.0;
    size_t cnt_core = 0;
    for (size_t k = idx_p20; k <= idx_p55; ++k) {
        sum_core += samples[k];
        ++cnt_core;
    }
    const double z_est = (cnt_core > 0) ? (sum_core / static_cast<double>(cnt_core)) : samples[idx_p35];
    const double iqr = std::max(0.0, samples[idx_p55] - samples[idx_p20]);
    const double sigma_triang = (z_est * z_est * disp_sigma) / std::max(1.0, fx * baseline);
    const double sigma_total = std::max(0.05, sigma_triang + 0.02 * z_est + 0.50 * iqr);

    out_depth_m = z_est;
    out_sigma_m = sigma_total;
    return true;
}

}  // namespace

FUSION_CPP_EXPORT int32_t cpp_recover_unmatched_detections(
    int32_t n_dets,
    const fusion_core_cpp::Detection2DC* dets,
    const uint8_t* matched_det_flags,
    int32_t n_existing,
    const fusion_core_cpp::ClusterInputC* existing_positions,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    const fusion_core_cpp::FrustumRecoveryConfigC* cfg,
    const fusion_core_cpp::StereoDepthImageC* stereo_depth,
    fusion_core_cpp::RecoveredConeC* out_recovered,
    int32_t max_recovered
) {
    if (n_dets <= 0 || !dets || !intr || !ext || !cfg || !out_recovered || max_recovered <= 0) {
        return 0;
    }

    const double fx = std::max(1e-6, intr->fx);
    const double fy = std::max(1e-6, intr->fy);
    const double* r = ext->r;
    const double* t = ext->t;
    const double cam_orig_x = -(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]);
    const double cam_orig_z = -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]);
    const double tan_slope = std::tan(cfg->ground_slope_rad);

    int32_t n_out = 0;
    for (int32_t idx = 0; idx < n_dets && n_out < max_recovered; ++idx) {
        if (matched_det_flags != nullptr && matched_det_flags[idx] != 0) {
            continue;
        }
        const auto& det = dets[idx];
        if (det.confidence < cfg->min_confidence) {
            continue;
        }
        const double h_px = std::max(0.0, det.ymax - det.ymin);
        if (h_px < cfg->min_bbox_height_px) {
            continue;
        }

        const double eff_height_m = (det.is_big_orange != 0) ? cfg->big_orange_height_m : cfg->cone_height_m;
        double depth_bbox = 0.0;
        if (intr->distortion_model == 1) {
            const double xd_mid = (0.5 * (det.xmin + det.xmax) - intr->cx) / fx;
            double xn_tmp = 0.0, yn_top = 0.0, yn_bot = 0.0;
            undistort_normalized_c(xd_mid, (det.ymin - intr->cy) / fy, intr, xn_tmp, yn_top);
            undistort_normalized_c(xd_mid, (det.ymax - intr->cy) / fy, intr, xn_tmp, yn_bot);
            depth_bbox = eff_height_m / std::max(1e-4, std::abs(yn_bot - yn_top));
        } else {
            depth_bbox = eff_height_m * fy / std::max(1e-3, h_px);
        }
        const double sigma_bbox = std::max(0.10, 0.14 * depth_bbox);

        double u = 0.5 * (det.xmin + det.xmax);
        double v = 0.5 * (det.ymin + det.ymax);
        if (det.polygon_count >= 3 && det.polygon_count <= 16) {
            double su = 0.0, sv = 0.0;
            for (int32_t k = 0; k < det.polygon_count; ++k) {
                su += det.polygon_u[k];
                sv += det.polygon_v[k];
            }
            u = su / static_cast<double>(det.polygon_count);
            v = sv / static_cast<double>(det.polygon_count);
        }

        const double xd = (u - intr->cx) / fx;
        const double yd = (v - intr->cy) / fy;
        double xn = xd, yn = yd;
        undistort_normalized_c(xd, yd, intr, xn, yn);

        // Inverse-variance fusion accumulator starting with bbox-height depth
        double inv_var_sum = 1.0 / (sigma_bbox * sigma_bbox);
        double weighted_depth_sum = depth_bbox * inv_var_sum;

        if (cfg->enable_ground_plane_depth_fusion != 0) {
            const double yd_bot = (det.ymax - intr->cy) / fy;
            double xn_b = xd, yn_b = yd_bot;
            undistort_normalized_c(xd, yd_bot, intr, xn_b, yn_b);
            const double dir_x = r[0] * xn_b + r[3] * yn_b + r[6];
            const double dir_z = r[2] * xn_b + r[5] * yn_b + r[8];
            const double denom_g = dir_z - dir_x * tan_slope;
            if (denom_g < -1e-3) {
                const double s_ground = (cfg->ground_z_base_m + cam_orig_x * tan_slope - cam_orig_z) / denom_g;
                if (s_ground >= cfg->min_depth_m && s_ground <= cfg->max_depth_m &&
                    std::abs(s_ground - depth_bbox) <= 0.35 * depth_bbox) {
                    const double sigma_gnd = std::max(0.08, 0.09 * s_ground);
                    const double w_gnd = 1.0 / (sigma_gnd * sigma_gnd);
                    inv_var_sum += w_gnd;
                    weighted_depth_sum += s_ground * w_gnd;
                }
            }
        }

        int32_t used_stereo = 0;
        if (cfg->enable_stereo_depth_fusion != 0 && stereo_depth != nullptr) {
            double z_stereo = 0.0, sigma_stereo = 0.0;
            if (sample_stereo_depth_for_det_c(det, stereo_depth, intr, z_stereo, sigma_stereo)) {
                const double prior_depth = weighted_depth_sum / inv_var_sum;
                if (z_stereo >= cfg->min_depth_m && z_stereo <= cfg->max_depth_m &&
                    std::abs(z_stereo - prior_depth) <= 0.45 * std::max(1.0, prior_depth)) {
                    const double w_stereo = 1.0 / (sigma_stereo * sigma_stereo);
                    inv_var_sum += w_stereo;
                    weighted_depth_sum += z_stereo * w_stereo;
                    used_stereo = 1;
                }
            }
        }

        const double depth = weighted_depth_sum / inv_var_sum;
        if (depth < cfg->min_depth_m || depth > cfg->max_depth_m) {
            continue;
        }
        const double sigma_r = std::max(0.12, std::min(cfg->range_std_fraction * depth, 1.0 / std::sqrt(inv_var_sum)));

        const double x_cam = xn * depth;
        const double y_cam = yn * depth;
        const double z_cam = depth;

        const double x0 = x_cam - t[0];
        const double y0 = y_cam - t[1];
        const double z0 = z_cam - t[2];
        const double x_l = r[0] * x0 + r[3] * y0 + r[6] * z0;
        const double y_l = r[1] * x0 + r[4] * y0 + r[7] * z0;
        const double z_l = r[2] * x0 + r[5] * y0 + r[8] * z0;

        bool too_close = false;
        const double dedup2 = cfg->dedup_radius_m * cfg->dedup_radius_m;
        if (existing_positions != nullptr) {
            for (int32_t e = 0; e < n_existing; ++e) {
                const double dx = x_l - existing_positions[e].cx;
                const double dy = y_l - existing_positions[e].cy;
                if (dx * dx + dy * dy < dedup2) {
                    too_close = true;
                    break;
                }
            }
        }
        if (too_close) continue;
        for (int32_t p = 0; p < n_out; ++p) {
            const double dx = x_l - out_recovered[p].x;
            const double dy = y_l - out_recovered[p].y;
            if (dx * dx + dy * dy < dedup2) {
                too_close = true;
                break;
            }
        }
        if (too_close) continue;

        const double psi = std::atan2(y_l, x_l);
        const double cp = std::cos(psi);
        const double sp = std::sin(psi);
        const double sigma_b = std::max(0.05, cfg->bearing_std_fraction * depth);
        const double var_r = sigma_r * sigma_r;
        const double var_b = sigma_b * sigma_b;

        auto& rec = out_recovered[n_out++];
        rec.x = x_l;
        rec.y = y_l;
        rec.z = z_l;
        rec.covariance[0] = cp * cp * var_r + sp * sp * var_b;
        rec.covariance[1] = cp * sp * (var_r - var_b);
        rec.covariance[2] = 0.0;
        rec.covariance[3] = rec.covariance[1];
        rec.covariance[4] = sp * sp * var_r + cp * cp * var_b;
        rec.covariance[5] = 0.0;
        rec.covariance[6] = 0.0;
        rec.covariance[7] = 0.0;
        rec.covariance[8] = std::max(0.10, 0.25 * var_r);
        normalize_probs4(det.class_probs, rec.class_probs);
        rec.confidence = det.confidence;
        rec.depth_m = depth;
        rec.detection_idx = idx;
        rec.used_stereo = used_stereo;
    }

    return n_out;
}

namespace {

struct TrackInternal {
    fusion_core_cpp::TrackEstimateC est{};
    double dirichlet_alpha[4]{0.25, 0.25, 0.25, 0.25};
};

struct GlobalLandmarkC {
    int32_t landmark_id = 0;
    double wx = 0.0;
    double wy = 0.0;
    double wz = 0.0;
    double cov_xx = 0.05;
    double cov_xy = 0.0;
    double cov_yy = 0.05;
    double dirichlet_alpha[4]{0.25, 0.25, 0.25, 0.25};
    double class_probs[4]{0.25, 0.25, 0.25, 0.25};
    int32_t observations_count = 0;
    double last_seen_stamp = 0.0;
};

struct GlobalLandmarkMapperC {
    double gate_radius_m = 1.5;
    double min_lap_distance_m = 40.0;
    double loop_closure_gain = 0.65;
    double pose_x = 0.0;
    double pose_y = 0.0;
    double pose_yaw = 0.0;
    double total_distance_m = 0.0;
    double last_loop_closure_dist_m = 0.0;
    int32_t loop_closures_count = 0;
    int32_t next_landmark_id = 1;
    std::vector<GlobalLandmarkC> landmarks;

    void reset(double gate_r, double min_lap) {
        gate_radius_m = std::max(0.3, gate_r);
        min_lap_distance_m = std::max(10.0, min_lap);
        pose_x = pose_y = pose_yaw = total_distance_m = last_loop_closure_dist_m = 0.0;
        loop_closures_count = 0;
        next_landmark_id = 1;
        landmarks.clear();
    }

    void step(
        const std::vector<TrackInternal>& tracks,
        double stamp,
        const fusion_core_cpp::EgoMotionC* ego,
        double dt
    ) {
        if (ego != nullptr && ego->has_motion != 0) {
            double dx = 0.0, dy = 0.0, dyaw = 0.0;
            if (ego->has_disp != 0) {
                dx = ego->dx;
                dy = ego->dy;
                dyaw = ego->dyaw;
            } else {
                const double step_dt = std::max(0.0, (ego->dt > 0.0) ? ego->dt : dt);
                dyaw = ego->wz * step_dt;
                if (std::abs(dyaw) > 1e-5) {
                    const double sd = std::sin(dyaw), cd = std::cos(dyaw);
                    dx = (sd * ego->vx - (1.0 - cd) * ego->vy) / ego->wz;
                    dy = ((1.0 - cd) * ego->vx + sd * ego->vy) / ego->wz;
                } else {
                    dx = ego->vx * step_dt - 0.5 * ego->vy * dyaw * step_dt;
                    dy = ego->vy * step_dt + 0.5 * ego->vx * dyaw * step_dt;
                }
            }
            const double cy = std::cos(pose_yaw), sy = std::sin(pose_yaw);
            pose_x += cy * dx - sy * dy;
            pose_y += sy * dx + cy * dy;
            pose_yaw = std::atan2(std::sin(pose_yaw + dyaw), std::cos(pose_yaw + dyaw));
            total_distance_m += std::hypot(dx, dy);
        }

        double cy = std::cos(pose_yaw), sy = std::sin(pose_yaw);
        const double gate_r2 = gate_radius_m * gate_radius_m;

        if ((total_distance_m - last_loop_closure_dist_m) >= min_lap_distance_m && landmarks.size() >= 4) {
            double sum_dx = 0.0, sum_dy = 0.0;
            int32_t cnt_lc = 0;
            const double lc_r2 = std::max(gate_r2, 2.5 * 2.5);
            for (const auto& t : tracks) {
                if (t.est.status != 1 || t.est.misses != 0) continue;
                const double bx = t.est.state[0], by = t.est.state[1];
                if (std::hypot(bx, by) > 15.0) continue;
                const double wx = pose_x + cy * bx - sy * by;
                const double wy = pose_y + sy * bx + cy * by;
                double tp[4];
                normalize_probs4(t.est.class_probs, tp);
                const GlobalLandmarkC* best_lm = nullptr;
                double best_d2 = lc_r2;
                for (const auto& lm : landmarks) {
                    if (lm.observations_count < 3 || (stamp - lm.last_seen_stamp) < 5.0) continue;
                    const double sem_sim = std::min(tp[0], lm.class_probs[0]) + std::min(tp[1], lm.class_probs[1]) +
                                           std::min(tp[2], lm.class_probs[2]) + std::min(tp[3], lm.class_probs[3]);
                    if (sem_sim < 0.45) continue;
                    const double d2 = (wx - lm.wx) * (wx - lm.wx) + (wy - lm.wy) * (wy - lm.wy);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_lm = &lm;
                    }
                }
                if (best_lm) {
                    sum_dx += best_lm->wx - wx;
                    sum_dy += best_lm->wy - wy;
                    ++cnt_lc;
                }
            }
            if (cnt_lc >= 2) {
                const double mdx = sum_dx / cnt_lc;
                const double mdy = sum_dy / cnt_lc;
                if (std::hypot(mdx, mdy) <= 3.0) {
                    pose_x += loop_closure_gain * mdx;
                    pose_y += loop_closure_gain * mdy;
                    last_loop_closure_dist_m = total_distance_m;
                    ++loop_closures_count;
                    cy = std::cos(pose_yaw);
                    sy = std::sin(pose_yaw);
                }
            }
        }

        for (const auto& t : tracks) {
            if (t.est.status != 1 || t.est.misses != 0) continue;
            const double bx = t.est.state[0], by = t.est.state[1];
            const double wx = pose_x + cy * bx - sy * by;
            const double wy = pose_y + sy * bx + cy * by;
            const double p00 = std::max(1e-4, t.est.covariance[0]);
            const double p01 = t.est.covariance[1];
            const double p11 = std::max(1e-4, t.est.covariance[5]);
            const double w_cov_xx = cy * (cy * p00 - sy * p01) - sy * (cy * p01 - sy * p11);
            const double w_cov_yy = sy * (sy * p00 + cy * p01) + cy * (sy * p01 + cy * p11);
            const double w_cov_xy = cy * (sy * p00 + cy * p01) - sy * (sy * p01 + cy * p11);
            double tp[4];
            normalize_probs4(t.est.class_probs, tp);

            int32_t best_idx = -1;
            double best_d2 = gate_r2;
            for (size_t i = 0; i < landmarks.size(); ++i) {
                const auto& lm = landmarks[i];
                const double d2 = (wx - lm.wx) * (wx - lm.wx) + (wy - lm.wy) * (wy - lm.wy);
                if (d2 < best_d2) {
                    const double sem_sim = std::min(tp[0], lm.class_probs[0]) + std::min(tp[1], lm.class_probs[1]) +
                                           std::min(tp[2], lm.class_probs[2]) + std::min(tp[3], lm.class_probs[3]);
                    if (sem_sim >= 0.20) {
                        best_d2 = d2;
                        best_idx = static_cast<int32_t>(i);
                    }
                }
            }

            if (best_idx >= 0) {
                auto& lm = landmarks[static_cast<size_t>(best_idx)];
                const double s00 = lm.cov_xx + w_cov_xx;
                const double s01 = lm.cov_xy + w_cov_xy;
                const double s11 = lm.cov_yy + w_cov_yy;
                const double det = s00 * s11 - s01 * s01;
                if (det > 1e-9) {
                    const double inv_d = 1.0 / det;
                    const double k00 = (lm.cov_xx * s11 - lm.cov_xy * s01) * inv_d;
                    const double k01 = (-lm.cov_xx * s01 + lm.cov_xy * s00) * inv_d;
                    const double k10 = (lm.cov_xy * s11 - lm.cov_yy * s01) * inv_d;
                    const double k11 = (-lm.cov_xy * s01 + lm.cov_yy * s00) * inv_d;
                    const double iy0 = wx - lm.wx, iy1 = wy - lm.wy;
                    lm.wx += k00 * iy0 + k01 * iy1;
                    lm.wy += k10 * iy0 + k11 * iy1;
                    lm.wz = 0.5 * (lm.wz + t.est.z);
                    const double nc_xx = std::max(1e-4, (1.0 - k00) * lm.cov_xx - k01 * lm.cov_xy);
                    const double nc_yy = std::max(1e-4, -k10 * lm.cov_xy + (1.0 - k11) * lm.cov_yy);
                    const double nc_xy = (1.0 - k00) * lm.cov_xy - k01 * lm.cov_yy;
                    lm.cov_xx = nc_xx;
                    lm.cov_yy = nc_yy;
                    lm.cov_xy = nc_xy;
                }
                const double info = normalized_information4(tp);
                if (info > 1e-3) {
                    for (int k = 0; k < 4; ++k) {
                        lm.dirichlet_alpha[k] = std::min(60.0, lm.dirichlet_alpha[k] + 1.5 * info * tp[k]);
                    }
                    normalize_probs4(lm.dirichlet_alpha, lm.class_probs);
                }
                ++lm.observations_count;
                lm.last_seen_stamp = stamp;
            } else {
                GlobalLandmarkC nlm;
                nlm.landmark_id = next_landmark_id++;
                nlm.wx = wx;
                nlm.wy = wy;
                nlm.wz = t.est.z;
                nlm.cov_xx = std::max(1e-3, w_cov_xx);
                nlm.cov_xy = w_cov_xy;
                nlm.cov_yy = std::max(1e-3, w_cov_yy);
                for (int k = 0; k < 4; ++k) {
                    nlm.dirichlet_alpha[k] = t.dirichlet_alpha[k];
                    nlm.class_probs[k] = tp[k];
                }
                nlm.observations_count = 1;
                nlm.last_seen_stamp = stamp;
                landmarks.push_back(nlm);
            }
        }
    }
};

struct TrackerInstance {
    fusion_core_cpp::TrackerConfigC cfg{};
    std::vector<TrackInternal> tracks;
    int32_t next_track_id = 1;
    double last_step_stamp = -1.0;
    GlobalLandmarkMapperC global_mapper;
    fusion_core_cpp::CorridorEstimateC last_corridor{};
    bool has_last_corridor = false;
    double last_corridor_stamp = -1e9;
};

struct TrackerSnapshotInternal {
    std::vector<TrackInternal> tracks;
    int32_t next_track_id = 1;
    double last_step_stamp = -1.0;
};

inline double infer_corridor_side_sign_c(
    const std::vector<double>& blue_ys,
    const std::vector<double>& yellow_ys
) {
    if (!blue_ys.empty() && !yellow_ys.empty()) {
        double sb = 0.0, sy = 0.0;
        for (double v : blue_ys) sb += v;
        for (double v : yellow_ys) sy += v;
        return (sb / blue_ys.size()) >= (sy / yellow_ys.size()) ? 1.0 : -1.0;
    }
    if (!blue_ys.empty()) {
        double sb = 0.0;
        for (double v : blue_ys) sb += v;
        return ((sb / blue_ys.size()) < -0.5) ? -1.0 : 1.0;
    }
    if (!yellow_ys.empty()) {
        double sy = 0.0;
        for (double v : yellow_ys) sy += v;
        return ((sy / yellow_ys.size()) > 0.5) ? -1.0 : 1.0;
    }
    return 1.0;
}

void build_corridor_estimate_c(
    const std::vector<TrackInternal>& tracks,
    double xm,
    double y0,
    double slope,
    double kappa,
    double confidence,
    double horizon_m,
    double step_m,
    double side_sign,
    fusion_core_cpp::CorridorEstimateC& out
) {
    double sum_left = 0.0, sum_right = 0.0;
    int32_t cnt_left = 0, cnt_right = 0;
    for (const auto& t : tracks) {
        if (t.est.misses <= 2 && t.est.state[0] >= 1.5 && t.est.state[0] <= horizon_m) {
            double tp[4];
            normalize_probs4(t.est.class_probs, tp);
            const double dx = t.est.state[0] - xm;
            const double yc = y0 + slope * dx + 0.5 * kappa * dx * dx;
            const double dy_signed = side_sign * (t.est.state[1] - yc);
            if (tp[0] > 0.50 && dy_signed > 0.5) {
                sum_left += dy_signed;
                ++cnt_left;
            } else if (tp[1] > 0.50 && dy_signed < -0.5) {
                sum_right += -dy_signed;
                ++cnt_right;
            }
        }
    }
    const double half_w_blue = (cnt_left > 0) ? clamp_d(sum_left / cnt_left, 1.25, 2.50) : 1.50;
    const double half_w_yellow = (cnt_right > 0) ? clamp_d(sum_right / cnt_right, 1.25, 2.50) : 1.50;
    const double half_w_pos_y = (side_sign >= 0.0) ? half_w_blue : half_w_yellow;
    const double half_w_neg_y = (side_sign >= 0.0) ? half_w_yellow : half_w_blue;

    const double eff_step = std::max(0.25, step_m);
    const int32_t n_steps = std::min<int32_t>(64, std::max<int32_t>(2, static_cast<int32_t>(std::llround(horizon_m / eff_step)) + 1));

    out.xm = xm;
    out.y0 = y0;
    out.slope = slope;
    out.curvature_kappa = kappa;
    out.track_width_m = half_w_blue + half_w_yellow;
    out.confidence = confidence;
    out.n_waypoints = n_steps;

    for (int32_t i = 0; i < n_steps; ++i) {
        const double x = std::min(horizon_m, static_cast<double>(i) * eff_step);
        const double dx = x - xm;
        const double yc = y0 + slope * dx + 0.5 * kappa * dx * dx;
        const double dy_dx = slope + kappa * dx;
        const double heading = std::atan(dy_dx);
        const double nx = -std::sin(heading);
        const double ny = std::cos(heading);
        out.left_x[i] = x + half_w_pos_y * nx;
        out.left_y[i] = yc + half_w_pos_y * ny;
        out.right_x[i] = x - half_w_neg_y * nx;
        out.right_y[i] = yc - half_w_neg_y * ny;
        out.wp_x[i] = x;
        out.wp_y[i] = yc;
        out.wp_heading[i] = heading;
        out.wp_kappa[i] = kappa;
    }
}

void regularize_observations_with_corridor_c(
    TrackerInstance* inst,
    int32_t n_obs,
    const fusion_core_cpp::Observation2DC* obs_in,
    const fusion_core_cpp::EgoMotionC* ego,
    std::vector<fusion_core_cpp::Observation2DC>& obs_out
) {
    obs_out.assign(obs_in, obs_in + n_obs);
    const double w_prior = clamp_d(inst->cfg.corridor_prior_weight, 0.0, 0.90);
    if (w_prior <= 1e-6 || n_obs <= 0) return;

    std::vector<std::pair<double, double>> blue_pts, yellow_pts;
    std::vector<double> blue_ys, yellow_ys;
    for (const auto& t : inst->tracks) {
        if (t.est.misses <= 2 && t.est.state[0] >= 1.8 && t.est.state[0] <= 15.5) {
            double tp[4];
            normalize_probs4(t.est.class_probs, tp);
            if (tp[0] > 0.55 && tp[2] < 0.30) {
                blue_pts.push_back({t.est.state[0], t.est.state[1]});
                blue_ys.push_back(t.est.state[1]);
            } else if (tp[1] > 0.55 && tp[2] < 0.30) {
                yellow_pts.push_back({t.est.state[0], t.est.state[1]});
                yellow_ys.push_back(t.est.state[1]);
            }
        }
    }
    for (int32_t i = 0; i < n_obs; ++i) {
        if (obs_in[i].x >= 1.8 && obs_in[i].x <= 15.0) {
            double op[4];
            normalize_probs4(obs_in[i].class_probs, op);
            if (op[0] > 0.65 && op[2] < 0.30) {
                blue_pts.push_back({obs_in[i].x, obs_in[i].y});
                blue_ys.push_back(obs_in[i].y);
            } else if (op[1] > 0.65 && op[2] < 0.30) {
                yellow_pts.push_back({obs_in[i].x, obs_in[i].y});
                yellow_ys.push_back(obs_in[i].y);
            }
        }
    }

    const double side_sign = infer_corridor_side_sign_c(blue_ys, yellow_ys);
    std::vector<double> xs_c, ys_c;
    xs_c.reserve(blue_pts.size() + yellow_pts.size());
    ys_c.reserve(blue_pts.size() + yellow_pts.size());
    for (const auto& p : blue_pts) {
        xs_c.push_back(p.first);
        ys_c.push_back(p.second - side_sign * 1.50);
    }
    for (const auto& p : yellow_pts) {
        xs_c.push_back(p.first);
        ys_c.push_back(p.second + side_sign * 1.50);
    }

    double xm = 6.0, y0 = 0.0, slope = 0.0, kappa = 0.0, fit_mse = 1.0;
    if (ego != nullptr && ego->has_motion != 0 && std::abs(ego->vx) > 1.5) {
        kappa = clamp_d(ego->wz / std::max(2.5, std::abs(ego->vx)), -0.06, 0.06);
    }
    if (xs_c.size() >= 2) {
        const size_t nc = xs_c.size();
        double sx = 0.0, sy = 0.0;
        double min_x = xs_c[0], max_x = xs_c[0];
        for (size_t i = 0; i < nc; ++i) {
            sx += xs_c[i];
            sy += ys_c[i];
            if (xs_c[i] < min_x) min_x = xs_c[i];
            if (xs_c[i] > max_x) max_x = xs_c[i];
        }
        xm = sx / static_cast<double>(nc);
        const double ym = clamp_d(sy / static_cast<double>(nc), -3.5, 3.5);
        y0 = ym;
        double var_x = 0.0, cov_xy = 0.0;
        for (size_t i = 0; i < nc; ++i) {
            const double u = xs_c[i] - xm;
            var_x += u * u;
            cov_xy += u * (ys_c[i] - ym);
        }
        if (var_x > 1.2) {
            slope = clamp_d(cov_xy / var_x, -0.45, 0.45);
            if (nc >= 4 && (max_x - min_x) >= 4.5) {
                const double u2_mean = var_x / static_cast<double>(nc);
                double var_v = 0.0, cov_vy = 0.0;
                for (size_t i = 0; i < nc; ++i) {
                    const double u = xs_c[i] - xm;
                    const double v = u * u - u2_mean;
                    var_v += v * v;
                    cov_vy += v * (ys_c[i] - ym - slope * u);
                }
                if (var_v > 12.0) {
                    const double k_fit = clamp_d(2.0 * (cov_vy / var_v), -0.06, 0.06);
                    kappa = (std::abs(kappa) > 1e-4) ? (0.75 * k_fit + 0.25 * kappa) : k_fit;
                    y0 = clamp_d(ym - 0.5 * kappa * u2_mean, -3.5, 3.5);
                }
            }
            double se = 0.0;
            for (size_t i = 0; i < nc; ++i) {
                const double u = xs_c[i] - xm;
                const double err = ys_c[i] - (y0 + slope * u + 0.5 * kappa * u * u);
                se += err * err;
            }
            fit_mse = se / static_cast<double>(nc);
        }
    }

    const bool well_cond = (blue_pts.size() >= 2 && yellow_pts.size() >= 2 && xs_c.size() >= 6 && fit_mse <= 0.06);
    const double conf = clamp_d(std::min(1.0, static_cast<double>(xs_c.size()) / 6.0) * std::exp(-3.0 * std::max(0.0, fit_mse)), 0.0, 1.0);
    build_corridor_estimate_c(inst->tracks, xm, y0, slope, kappa, conf, 20.0, 1.0, side_sign, inst->last_corridor);
    inst->has_last_corridor = true;
    inst->last_corridor_stamp = inst->last_step_stamp;

    for (int32_t i = 0; i < n_obs; ++i) {
        double p[4];
        normalize_probs4(obs_out[static_cast<size_t>(i)].class_probs, p);
        if (p[2] >= 0.35) continue;
        const double dx_m = obs_out[static_cast<size_t>(i)].x - xm;
        const double yc = y0 + slope * dx_m + 0.5 * kappa * dx_m * dx_m;
        const double dy_signed = side_sign * (obs_out[static_cast<size_t>(i)].y - yc);
        if (std::abs(dy_signed) < 0.45) continue;
        const double p_blue_corr = 1.0 / (1.0 + std::exp(-clamp_d(2.2 * dy_signed, -12.0, 12.0)));
        const double p_yellow_corr = 1.0 - p_blue_corr;
        const double r = std::hypot(obs_out[static_cast<size_t>(i)].x, obs_out[static_cast<size_t>(i)].y);
        const double info = normalized_information4(p);
        const bool contradicts_far = well_cond && r >= 14.0 &&
            ((dy_signed > 1.20 && p[1] > p[0]) || (dy_signed < -1.20 && p[0] > p[1]));
        if ((info >= 0.03 && info <= 0.45 && r >= 8.5) || contradicts_far) {
            const double w_eff = contradicts_far
                ? clamp_d(std::max(w_prior, 0.72) * std::min(1.0, (r - 10.0) / 5.0), 0.0, 0.85)
                : clamp_d(w_prior * std::min(1.0, (r - 7.0) / 8.0), 0.0, 0.80);
            const double by_mass = std::max(0.50, p[0] + p[1]);
            const double raw[4] = {
                (1.0 - w_eff) * p[0] + w_eff * by_mass * p_blue_corr,
                (1.0 - w_eff) * p[1] + w_eff * by_mass * p_yellow_corr,
                p[2],
                p[3]
            };
            normalize_probs4(raw, obs_out[static_cast<size_t>(i)].class_probs);
        }
    }
}

inline void mat_mul_4x4(const double a[16], const double b[16], double out[16]) {
    for (int r = 0; r < 4; ++r) {
        const int r4 = r * 4;
        for (int c = 0; c < 4; ++c) {
            out[r4 + c] = a[r4 + 0] * b[0 + c] + a[r4 + 1] * b[4 + c] +
                          a[r4 + 2] * b[8 + c] + a[r4 + 3] * b[12 + c];
        }
    }
}

inline void mat_mul_4x4_trans_b(const double a[16], const double b[16], double out[16]) {
    for (int r = 0; r < 4; ++r) {
        const int r4 = r * 4;
        for (int c = 0; c < 4; ++c) {
            const int c4 = c * 4;
            out[r4 + c] = a[r4 + 0] * b[c4 + 0] + a[r4 + 1] * b[c4 + 1] +
                          a[r4 + 2] * b[c4 + 2] + a[r4 + 3] * b[c4 + 3];
        }
    }
}

void predict_track_c(
    TrackInternal& tr,
    double stamp,
    double prev_step_stamp,
    const fusion_core_cpp::EgoMotionC* ego,
    const fusion_core_cpp::TrackerConfigC& cfg
) {
    const double ref_stamp = (prev_step_stamp >= 0.0)
                                 ? std::max(tr.est.last_stamp, prev_step_stamp)
                                 : tr.est.last_stamp;
    const double dt = clamp_d(stamp - ref_stamp, 0.0, cfg.max_predict_dt);
    if (dt <= 0.0) {
        return;
    }

    if (cfg.motion_model_landmark != 0) {
        const double x = tr.est.state[0];
        const double y = tr.est.state[1];
        double vx_ego = 0.0, vy_ego = 0.0, wz_ego = 0.0;
        double dx = 0.0, dy = 0.0, dtheta = 0.0;
        const bool has_ego = (ego != nullptr && ego->has_motion != 0);
        if (has_ego) {
            vx_ego = ego->vx;
            vy_ego = ego->vy;
            wz_ego = ego->wz;
            if (ego->has_disp != 0) {
                dx = ego->dx;
                dy = ego->dy;
                dtheta = ego->dyaw;
            } else {
                const double step_dt = clamp_d((ego->dt > 0.0) ? ego->dt : dt, 0.0, cfg.max_predict_dt);
                dtheta = wz_ego * step_dt;
                if (std::abs(dtheta) > 1e-5) {
                    const double sin_dt = std::sin(dtheta);
                    const double cos_dt = std::cos(dtheta);
                    dx = (sin_dt * vx_ego - (1.0 - cos_dt) * vy_ego) / wz_ego;
                    dy = ((1.0 - cos_dt) * vx_ego + sin_dt * vy_ego) / wz_ego;
                } else {
                    dx = vx_ego * step_dt - 0.5 * vy_ego * dtheta * step_dt;
                    dy = vy_ego * step_dt + 0.5 * vx_ego * dtheta * step_dt;
                }
            }
        }

        const double cos_t = std::cos(dtheta);
        const double sin_t = std::sin(dtheta);
        const double xt = x - dx;
        const double yt = y - dy;
        const double x_new = cos_t * xt + sin_t * yt;
        const double y_new = -sin_t * xt + cos_t * yt;
        tr.est.state[0] = x_new;
        tr.est.state[1] = y_new;
        tr.est.state[2] = -vx_ego + wz_ego * y_new;
        tr.est.state[3] = -vy_ego - wz_ego * x_new;

        const double p00 = tr.est.covariance[0];
        const double p01 = tr.est.covariance[1];
        const double p10 = tr.est.covariance[4];
        const double p11 = tr.est.covariance[5];

        const double rp00 = cos_t * p00 + sin_t * p10;
        const double rp01 = cos_t * p01 + sin_t * p11;
        const double rp10 = -sin_t * p00 + cos_t * p10;
        const double rp11 = -sin_t * p01 + cos_t * p11;

        const double rpr00 = rp00 * cos_t + rp01 * sin_t;
        const double rpr01 = -rp00 * sin_t + rp01 * cos_t;
        const double rpr10 = rp10 * cos_t + rp11 * sin_t;
        const double rpr11 = -rp10 * sin_t + rp11 * cos_t;

        const double q_pos = (cfg.landmark_process_noise_pos * cfg.landmark_process_noise_pos) * dt;
        double q00 = 0.0, q01 = 0.0, q11 = 0.0;
        if (has_ego) {
            const double var_dx = (cfg.odom_vx_std * dt) * (cfg.odom_vx_std * dt);
            const double var_dy = (cfg.odom_vy_std * dt) * (cfg.odom_vy_std * dt);
            const double var_dtheta = (cfg.odom_wz_std * dt) * (cfg.odom_wz_std * dt);
            q00 = q_pos + var_dx + (y_new * y_new) * var_dtheta;
            q11 = q_pos + var_dy + (x_new * x_new) * var_dtheta;
            q01 = -(x_new * y_new) * var_dtheta;
        } else {
            const double q_static = std::max(q_pos, 0.25 * (cfg.process_noise_accel * cfg.process_noise_accel) * (dt * dt * dt));
            q00 = q_static;
            q11 = q_static;
            q01 = 0.0;
        }

        const double p22 = tr.est.covariance[10];
        const double p33 = tr.est.covariance[15];
        for (int i = 0; i < 16; ++i) tr.est.covariance[i] = 0.0;
        tr.est.covariance[0] = rpr00 + q00;
        tr.est.covariance[1] = 0.5 * (rpr01 + rpr10) + q01;
        tr.est.covariance[4] = tr.est.covariance[1];
        tr.est.covariance[5] = rpr11 + q11;
        tr.est.covariance[10] = p22;
        tr.est.covariance[15] = p33;
        return;
    }

    // Legacy Constant Velocity (CV)
    tr.est.state[0] += tr.est.state[2] * dt;
    tr.est.state[1] += tr.est.state[3] * dt;

    double f[16] = {
        1.0, 0.0, dt,  0.0,
        0.0, 1.0, 0.0, dt,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    const double sigma_a2 = cfg.process_noise_accel * cfg.process_noise_accel;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    const double q[16] = {
        0.25 * dt4 * sigma_a2, 0.0, 0.5 * dt3 * sigma_a2, 0.0,
        0.0, 0.25 * dt4 * sigma_a2, 0.0, 0.5 * dt3 * sigma_a2,
        0.5 * dt3 * sigma_a2, 0.0, dt2 * sigma_a2, 0.0,
        0.0, 0.5 * dt3 * sigma_a2, 0.0, dt2 * sigma_a2
    };
    double fp[16], fpf_t[16];
    mat_mul_4x4(f, tr.est.covariance, fp);
    mat_mul_4x4_trans_b(fp, f, fpf_t);
    for (int i = 0; i < 16; ++i) {
        tr.est.covariance[i] = fpf_t[i] + q[i];
    }
}

void update_track_c(
    TrackInternal& tr,
    const fusion_core_cpp::Observation2DC& obs,
    double stamp,
    double d2,
    const fusion_core_cpp::EgoMotionC* ego,
    const fusion_core_cpp::TrackerConfigC& cfg
) {
    const double* p = tr.est.covariance;
    const double r00 = std::max(cfg.min_meas_var, obs.cov_xx);
    const double r11 = std::max(cfg.min_meas_var, obs.cov_yy);
    const double r01 = obs.cov_xy;

    const double s00 = p[0] + r00;
    const double s01 = p[1] + r01;
    const double s10 = p[4] + r01;
    const double s11 = p[5] + r11;
    const double det = s00 * s11 - s01 * s10;
    if (std::abs(det) < 1e-12) {
        return;
    }
    const double inv_det = 1.0 / det;
    const double inv_s[4] = {s11 * inv_det, -s01 * inv_det, -s10 * inv_det, s00 * inv_det};

    double k[8];  // 4x2
    for (int row = 0; row < 4; ++row) {
        const double ph0 = p[row * 4 + 0];
        const double ph1 = p[row * 4 + 1];
        k[row * 2 + 0] = ph0 * inv_s[0] + ph1 * inv_s[2];
        k[row * 2 + 1] = ph0 * inv_s[1] + ph1 * inv_s[3];
    }

    const double y0 = obs.x - tr.est.state[0];
    const double y1 = obs.y - tr.est.state[1];
    for (int row = 0; row < 4; ++row) {
        tr.est.state[row] += k[row * 2 + 0] * y0 + k[row * 2 + 1] * y1;
    }

    if (cfg.motion_model_landmark != 0) {
        if (ego != nullptr && ego->has_motion != 0) {
            tr.est.state[2] = -ego->vx + ego->wz * tr.est.state[1];
            tr.est.state[3] = -ego->vy - ego->wz * tr.est.state[0];
        } else {
            tr.est.state[2] = 0.0;
            tr.est.state[3] = 0.0;
        }
    }

    // Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
    double imkh[16] = {
        1.0 - k[0], -k[1], 0.0, 0.0,
        -k[2], 1.0 - k[3], 0.0, 0.0,
        -k[4], -k[5],      1.0, 0.0,
        -k[6], -k[7],      0.0, 1.0
    };
    double tmp1[16], term1[16];
    mat_mul_4x4(imkh, tr.est.covariance, tmp1);
    mat_mul_4x4_trans_b(tmp1, imkh, term1);

    double kr[8];
    for (int row = 0; row < 4; ++row) {
        kr[row * 2 + 0] = k[row * 2 + 0] * r00 + k[row * 2 + 1] * r01;
        kr[row * 2 + 1] = k[row * 2 + 0] * r01 + k[row * 2 + 1] * r11;
    }
    double p_next[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            const double krkt = kr[r * 2 + 0] * k[c * 2 + 0] + kr[r * 2 + 1] * k[c * 2 + 1];
            p_next[r * 4 + c] = term1[r * 4 + c] + krkt;
        }
    }
    for (int r = 0; r < 4; ++r) {
        for (int c = r; c < 4; ++c) {
            const double sym = 0.5 * (p_next[r * 4 + c] + p_next[c * 4 + r]);
            tr.est.covariance[r * 4 + c] = sym;
            tr.est.covariance[c * 4 + r] = sym;
        }
    }

    const double innovation_quality = std::exp(-0.5 * std::min(25.0, std::max(0.0, d2)));
    double class_obs[4];
    normalize_probs4(obs.class_probs, class_obs);
    if (cfg.semantic_dirichlet != 0) {
        const double info = normalized_information4(class_obs);
        if (info > 1e-3) {
            const double ew = cfg.dirichlet_evidence_scale *
                              clamp_d(obs.quality, 0.05, 1.0) *
                              std::max(0.2, innovation_quality) * info;
            double total_conc = 0.0;
            for (int i = 0; i < 4; ++i) {
                tr.dirichlet_alpha[i] += ew * class_obs[i];
                total_conc += tr.dirichlet_alpha[i];
            }
            if (total_conc > cfg.dirichlet_max_concentration && total_conc > 1e-9) {
                const double scale = cfg.dirichlet_max_concentration / total_conc;
                for (int i = 0; i < 4; ++i) tr.dirichlet_alpha[i] *= scale;
                total_conc = cfg.dirichlet_max_concentration;
            }
            tr.est.semantic_certainty = total_conc;
            normalize_probs4(tr.dirichlet_alpha, tr.est.class_probs);
        }
    } else {
        const double alpha = clamp_d(cfg.class_update_alpha, 0.0, 1.0);
        double class_tr[4];
        normalize_probs4(tr.est.class_probs, class_tr);
        double raw[4];
        for (int i = 0; i < 4; ++i) {
            raw[i] = alpha * class_obs[i] + (1.0 - alpha) * class_tr[i];
        }
        normalize_probs4(raw, tr.est.class_probs);
    }

    const double obs_z_var = std::max(cfg.min_meas_var, obs.z_var);
    const double trk_z_var = std::max(cfg.min_meas_var, tr.est.z_var);
    const double denom_z = obs_z_var + trk_z_var;
    tr.est.z = (trk_z_var * obs.z + obs_z_var * tr.est.z) / denom_z;
    tr.est.z_var = std::max(cfg.min_meas_var, (obs_z_var * trk_z_var) / denom_z);

    tr.est.hits += 1;
    tr.est.misses = 0;
    tr.est.last_stamp = stamp;
    tr.est.status = (tr.est.hits >= cfg.confirm_hits) ? 1 : 0;
    tr.est.quality = clamp_d(
        0.5 * tr.est.quality + 0.5 * clamp_d(obs.quality, 0.0, 1.0) * innovation_quality,
        0.0, 1.0
    );
}

}  // namespace

FUSION_CPP_EXPORT void* cpp_tracker_create(const fusion_core_cpp::TrackerConfigC* cfg) {
    if (!cfg) return nullptr;
    auto* inst = new TrackerInstance();
    inst->cfg = *cfg;
    inst->global_mapper.reset(cfg->global_map_gate_radius_m, cfg->global_map_min_lap_distance_m);
    return static_cast<void*>(inst);
}

FUSION_CPP_EXPORT void cpp_tracker_destroy(void* handle) {
    if (!handle) return;
    delete static_cast<TrackerInstance*>(handle);
}

FUSION_CPP_EXPORT void cpp_tracker_reset(void* handle) {
    if (!handle) return;
    auto* inst = static_cast<TrackerInstance*>(handle);
    inst->tracks.clear();
    inst->next_track_id = 1;
    inst->last_step_stamp = -1.0;
    inst->has_last_corridor = false;
    inst->last_corridor_stamp = -1e9;
    inst->global_mapper.reset(inst->cfg.global_map_gate_radius_m, inst->cfg.global_map_min_lap_distance_m);
}

FUSION_CPP_EXPORT int32_t cpp_tracker_get_tracks(
    const void* handle,
    fusion_core_cpp::TrackEstimateC* out_tracks,
    int32_t max_tracks
) {
    if (!handle || !out_tracks || max_tracks <= 0) return 0;
    const auto* inst = static_cast<const TrackerInstance*>(handle);
    const int32_t n = std::min(static_cast<int32_t>(inst->tracks.size()), max_tracks);
    for (int32_t i = 0; i < n; ++i) {
        out_tracks[i] = inst->tracks[static_cast<size_t>(i)].est;
    }
    return n;
}

FUSION_CPP_EXPORT int32_t cpp_tracker_step(
    void* handle,
    double stamp,
    const fusion_core_cpp::EgoMotionC* ego,
    int32_t n_obs,
    const fusion_core_cpp::Observation2DC* obs,
    fusion_core_cpp::TrackEstimateC* out_tracks,
    int32_t max_tracks,
    fusion_core_cpp::TrackEstimateC* out_lost_tracks,
    int32_t max_lost_tracks,
    fusion_core_cpp::TrackerStepStatsC* out_stats
) {
    if (!handle) return 0;
    auto* inst = static_cast<TrackerInstance*>(handle);
    const auto& cfg = inst->cfg;

    fusion_core_cpp::TrackerStepStatsC stats{};
    if (stamp < inst->last_step_stamp) {
        if (out_stats) *out_stats = stats;
        return cpp_tracker_get_tracks(handle, out_tracks, max_tracks);
    }

    const double prev_step_stamp = inst->last_step_stamp;
    inst->last_step_stamp = stamp;

    for (auto& tr : inst->tracks) {
        predict_track_c(tr, stamp, prev_step_stamp, ego, cfg);
    }

    std::vector<fusion_core_cpp::Observation2DC> reg_obs_buf;
    const fusion_core_cpp::Observation2DC* eff_obs = obs;
    if (cfg.corridor_prior_weight > 0.0 && n_obs > 0 && obs != nullptr) {
        regularize_observations_with_corridor_c(inst, n_obs, obs, ego, reg_obs_buf);
        eff_obs = reg_obs_buf.data();
    }

    const int32_t n_tracks = static_cast<int32_t>(inst->tracks.size());
    struct CandidateC {
        double cost;
        int32_t tr_idx;
        int32_t obs_idx;
        double d2;
    };
    std::vector<CandidateC> candidates;
    if (n_tracks > 0 && n_obs > 0 && eff_obs != nullptr) {
        std::vector<double> obs_probs(static_cast<size_t>(n_obs) * 4);
        std::vector<double> obs_r00(static_cast<size_t>(n_obs));
        std::vector<double> obs_r11(static_cast<size_t>(n_obs));
        for (int32_t o = 0; o < n_obs; ++o) {
            normalize_probs4(eff_obs[o].class_probs, &obs_probs[static_cast<size_t>(o) * 4]);
            obs_r00[static_cast<size_t>(o)] = std::max(cfg.min_meas_var, eff_obs[o].cov_xx);
            obs_r11[static_cast<size_t>(o)] = std::max(cfg.min_meas_var, eff_obs[o].cov_yy);
        }

        for (int32_t t = 0; t < n_tracks; ++t) {
            const auto& tr = inst->tracks[static_cast<size_t>(t)].est;
            const double tx = tr.state[0], ty = tr.state[1];
            const double p00 = tr.covariance[0], p01 = tr.covariance[1];
            const double p10 = tr.covariance[4], p11 = tr.covariance[5];
            double tp[4];
            normalize_probs4(tr.class_probs, tp);

            for (int32_t o = 0; o < n_obs; ++o) {
                const double y0 = eff_obs[o].x - tx;
                const double y1 = eff_obs[o].y - ty;
                const double s00 = p00 + obs_r00[static_cast<size_t>(o)];
                const double s11 = p11 + obs_r11[static_cast<size_t>(o)];
                const double s01 = p01 + eff_obs[o].cov_xy;
                const double s10 = p10 + eff_obs[o].cov_xy;
                const double det = s00 * s11 - s01 * s10;
                if (std::abs(det) < 1e-9) continue;
                double d2 = (y0 * (s11 * y0 - s01 * y1) + y1 * (-s10 * y0 + s00 * y1)) / det;
                if (d2 < 0.0) d2 = 0.0;
                if (d2 <= cfg.mahalanobis_gate) {
                    const double* op = &obs_probs[static_cast<size_t>(o) * 4];
                    const double sem_ov = std::min(tp[0], op[0]) + std::min(tp[1], op[1]) +
                                          std::min(tp[2], op[2]) + std::min(tp[3], op[3]);
                    if (sem_ov >= cfg.semantic_min_overlap) {
                        const double cost = d2 + cfg.semantic_cost_weight * (1.0 - sem_ov);
                        candidates.push_back({cost, t, o, d2});
                        ++stats.assoc_candidates;
                    }
                }
            }
        }
    }

    struct MatchC {
        int32_t tr_idx;
        int32_t obs_idx;
        double d2;
    };
    std::vector<MatchC> matches;
    std::vector<uint8_t> track_matched(static_cast<size_t>(n_tracks), 0);
    std::vector<uint8_t> obs_matched(static_cast<size_t>(std::max(0, n_obs)), 0);

    if (!candidates.empty()) {
        if (cfg.assignment_hungarian != 0 && n_tracks > 1 && n_obs > 1) {
            constexpr double kDisallowed = 1e6;
            const size_t n_cells = static_cast<size_t>(n_tracks) * static_cast<size_t>(n_obs);
            std::vector<double> cost_mat(n_cells, kDisallowed);
            std::vector<double> d2_mat(n_cells, 0.0);
            std::vector<uint8_t> valid_mat(n_cells, 0);
            for (const auto& c : candidates) {
                const size_t idx = static_cast<size_t>(c.tr_idx) * static_cast<size_t>(n_obs) +
                                   static_cast<size_t>(c.obs_idx);
                cost_mat[idx] = c.cost;
                d2_mat[idx] = c.d2;
                valid_mat[idx] = 1;
            }
            auto sol = solve_hungarian_min_cost(n_tracks, n_obs, cost_mat);
            for (int32_t t = 0; t < n_tracks; ++t) {
                const int32_t o = sol[static_cast<size_t>(t)];
                if (o >= 0 && o < n_obs) {
                    const size_t idx = static_cast<size_t>(t) * static_cast<size_t>(n_obs) +
                                       static_cast<size_t>(o);
                    if (valid_mat[idx] != 0) {
                        matches.push_back({t, o, d2_mat[idx]});
                        track_matched[static_cast<size_t>(t)] = 1;
                        obs_matched[static_cast<size_t>(o)] = 1;
                    }
                }
            }
        } else {
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const CandidateC& a, const CandidateC& b) {
                                 return a.cost < b.cost;
                             });
            for (const auto& c : candidates) {
                if (track_matched[static_cast<size_t>(c.tr_idx)] != 0 ||
                    obs_matched[static_cast<size_t>(c.obs_idx)] != 0) {
                    continue;
                }
                track_matched[static_cast<size_t>(c.tr_idx)] = 1;
                obs_matched[static_cast<size_t>(c.obs_idx)] = 1;
                matches.push_back({c.tr_idx, c.obs_idx, c.d2});
            }
        }
    }

    stats.assoc_matches = static_cast<int32_t>(matches.size());
    for (const auto& m : matches) {
        update_track_c(inst->tracks[static_cast<size_t>(m.tr_idx)],
                       eff_obs[m.obs_idx], stamp, m.d2, ego, cfg);
    }

    std::vector<TrackInternal> surviving;
    surviving.reserve(inst->tracks.size());
    int32_t lost_count = 0;
    for (int32_t t = 0; t < n_tracks; ++t) {
        auto tr = inst->tracks[static_cast<size_t>(t)];
        if (track_matched[static_cast<size_t>(t)] == 0) {
            tr.est.misses += 1;
            tr.est.quality = clamp_d(tr.est.quality * 0.92, 0.0, 1.0);
            if (tr.est.misses > cfg.max_missed) {
                tr.est.status = 2;
                if (out_lost_tracks != nullptr && lost_count < max_lost_tracks) {
                    out_lost_tracks[lost_count] = tr.est;
                }
                ++lost_count;
                ++stats.tracks_deleted;
                continue;
            }
            ++stats.assoc_rejects;
        }
        surviving.push_back(tr);
    }
    stats.n_lost_tracks = lost_count;
    inst->tracks.swap(surviving);

    const double vel_var = (cfg.motion_model_landmark != 0) ? 0.25 : 2.0;
    const double prior_base = std::max(0.05, 0.25 * cfg.dirichlet_prior_strength);
    for (int32_t o = 0; o < n_obs; ++o) {
        if (obs_matched[static_cast<size_t>(o)] != 0) continue;
        TrackInternal nt{};
        nt.est.track_id = inst->next_track_id++;
        double vx_init = 0.0, vy_init = 0.0;
        if (cfg.motion_model_landmark != 0 && ego != nullptr && ego->has_motion != 0) {
            vx_init = -ego->vx + ego->wz * eff_obs[o].y;
            vy_init = -ego->vy - ego->wz * eff_obs[o].x;
        }
        nt.est.state[0] = eff_obs[o].x;
        nt.est.state[1] = eff_obs[o].y;
        nt.est.state[2] = vx_init;
        nt.est.state[3] = vy_init;
        nt.est.covariance[0] = std::max(cfg.min_meas_var, eff_obs[o].cov_xx);
        nt.est.covariance[1] = eff_obs[o].cov_xy;
        nt.est.covariance[4] = eff_obs[o].cov_xy;
        nt.est.covariance[5] = std::max(cfg.min_meas_var, eff_obs[o].cov_yy);
        nt.est.covariance[10] = vel_var;
        nt.est.covariance[15] = vel_var;
        nt.est.z = eff_obs[o].z;
        nt.est.z_var = std::max(cfg.min_meas_var, eff_obs[o].z_var);

        double norm_p[4];
        normalize_probs4(eff_obs[o].class_probs, norm_p);
        const double info = normalized_information4(norm_p);
        const double init_w = cfg.dirichlet_evidence_scale * clamp_d(eff_obs[o].quality, 0.1, 1.0) * info;
        double sum_a = 0.0;
        for (int k = 0; k < 4; ++k) {
            nt.est.class_probs[k] = norm_p[k];
            nt.dirichlet_alpha[k] = prior_base + init_w * norm_p[k];
            sum_a += nt.dirichlet_alpha[k];
        }
        nt.est.hits = 1;
        nt.est.misses = 0;
        nt.est.quality = clamp_d(eff_obs[o].quality, 0.0, 1.0);
        nt.est.status = 0;
        nt.est.first_stamp = stamp;
        nt.est.last_stamp = stamp;
        nt.est.semantic_certainty = sum_a;
        inst->tracks.push_back(nt);
        ++stats.tracks_created;
    }

    if (cfg.enable_global_map != 0) {
        const double dt_step = (prev_step_stamp >= 0.0) ? std::max(0.0, stamp - prev_step_stamp) : 0.0;
        inst->global_mapper.step(inst->tracks, stamp, ego, dt_step);
    }

    if (out_stats) *out_stats = stats;
    return cpp_tracker_get_tracks(handle, out_tracks, max_tracks);
}

FUSION_CPP_EXPORT void* cpp_tracker_snapshot_create(const void* handle) {
    if (!handle) return nullptr;
    const auto* inst = static_cast<const TrackerInstance*>(handle);
    auto* snap = new TrackerSnapshotInternal();
    snap->tracks = inst->tracks;
    snap->next_track_id = inst->next_track_id;
    snap->last_step_stamp = inst->last_step_stamp;
    return static_cast<void*>(snap);
}

FUSION_CPP_EXPORT void cpp_tracker_snapshot_restore(void* handle, const void* snapshot) {
    if (!handle || !snapshot) return;
    auto* inst = static_cast<TrackerInstance*>(handle);
    const auto* snap = static_cast<const TrackerSnapshotInternal*>(snapshot);
    inst->tracks = snap->tracks;
    inst->next_track_id = snap->next_track_id;
    inst->last_step_stamp = snap->last_step_stamp;
    inst->last_corridor_stamp = -1.0;
}

FUSION_CPP_EXPORT void cpp_tracker_snapshot_destroy(void* snapshot) {
    if (!snapshot) return;
    delete static_cast<TrackerSnapshotInternal*>(snapshot);
}

FUSION_CPP_EXPORT double cpp_tracker_get_last_stamp(const void* handle) {
    if (!handle) return -1.0;
    return static_cast<const TrackerInstance*>(handle)->last_step_stamp;
}

FUSION_CPP_EXPORT int32_t cpp_tracker_get_corridor(
    const void* handle,
    double horizon_m,
    double step_m,
    const fusion_core_cpp::EgoMotionC* ego,
    fusion_core_cpp::CorridorEstimateC* out_corridor
) {
    if (!handle || !out_corridor) return 0;
    const auto* inst = static_cast<const TrackerInstance*>(handle);
    if (inst->has_last_corridor &&
        std::abs(inst->last_corridor_stamp - inst->last_step_stamp) < 1e-9 &&
        std::abs(horizon_m - 20.0) < 1e-3 &&
        std::abs(step_m - 1.0) < 1e-3) {
        *out_corridor = inst->last_corridor;
        return 1;
    }

    const int32_t min_hits_pref = std::min<int32_t>(2, std::max<int32_t>(1, inst->cfg.confirm_hits));
    std::vector<const TrackInternal*> eligible;
    for (const auto& t : inst->tracks) {
        if (t.est.misses <= 2 && t.est.state[0] >= 1.5 && t.est.state[0] <= 16.0 && t.est.hits >= min_hits_pref) {
            eligible.push_back(&t);
        }
    }
    if (eligible.size() < 2) {
        eligible.clear();
        for (const auto& t : inst->tracks) {
            if (t.est.misses <= 2 && t.est.state[0] >= 1.5 && t.est.state[0] <= 16.0) {
                eligible.push_back(&t);
            }
        }
    }

    std::vector<std::pair<double, double>> blue_pts, yellow_pts;
    std::vector<double> blue_ys, yellow_ys;
    for (const auto* tp_ptr : eligible) {
        double tp[4];
        normalize_probs4(tp_ptr->est.class_probs, tp);
        if (tp[0] > 0.50 && tp[2] < 0.35) {
            blue_pts.push_back({tp_ptr->est.state[0], tp_ptr->est.state[1]});
            blue_ys.push_back(tp_ptr->est.state[1]);
        } else if (tp[1] > 0.50 && tp[2] < 0.35) {
            yellow_pts.push_back({tp_ptr->est.state[0], tp_ptr->est.state[1]});
            yellow_ys.push_back(tp_ptr->est.state[1]);
        }
    }

    const double side_sign = infer_corridor_side_sign_c(blue_ys, yellow_ys);
    std::vector<double> xs_c, ys_c;
    for (const auto& p : blue_pts) {
        xs_c.push_back(p.first);
        ys_c.push_back(p.second - side_sign * 1.50);
    }
    for (const auto& p : yellow_pts) {
        xs_c.push_back(p.first);
        ys_c.push_back(p.second + side_sign * 1.50);
    }

    double xm = 6.0, y0 = 0.0, slope = 0.0, kappa = 0.0, fit_mse = 1.0;
    if (ego != nullptr && ego->has_motion != 0 && std::abs(ego->vx) > 1.5) {
        kappa = clamp_d(ego->wz / std::max(2.5, std::abs(ego->vx)), -0.06, 0.06);
    }
    if (xs_c.size() >= 2) {
        const size_t nc = xs_c.size();
        double sx = 0.0, sy = 0.0, min_x = xs_c[0], max_x = xs_c[0];
        for (size_t i = 0; i < nc; ++i) {
            sx += xs_c[i];
            sy += ys_c[i];
            if (xs_c[i] < min_x) min_x = xs_c[i];
            if (xs_c[i] > max_x) max_x = xs_c[i];
        }
        xm = sx / static_cast<double>(nc);
        const double ym = clamp_d(sy / static_cast<double>(nc), -3.5, 3.5);
        y0 = ym;
        double var_x = 0.0, cov_xy = 0.0;
        for (size_t i = 0; i < nc; ++i) {
            const double u = xs_c[i] - xm;
            var_x += u * u;
            cov_xy += u * (ys_c[i] - ym);
        }
        if (var_x > 1.2) {
            slope = clamp_d(cov_xy / var_x, -0.45, 0.45);
            if (nc >= 4 && (max_x - min_x) >= 4.0) {
                const double u2_mean = var_x / static_cast<double>(nc);
                double var_v = 0.0, cov_vy = 0.0;
                for (size_t i = 0; i < nc; ++i) {
                    const double u = xs_c[i] - xm;
                    const double v = u * u - u2_mean;
                    var_v += v * v;
                    cov_vy += v * (ys_c[i] - ym - slope * u);
                }
                if (var_v > 10.0) {
                    const double k_fit = clamp_d(2.0 * (cov_vy / var_v), -0.06, 0.06);
                    kappa = (std::abs(kappa) > 1e-4) ? (0.75 * k_fit + 0.25 * kappa) : k_fit;
                    y0 = clamp_d(ym - 0.5 * kappa * u2_mean, -3.5, 3.5);
                }
            }
            double se = 0.0;
            for (size_t i = 0; i < nc; ++i) {
                const double u = xs_c[i] - xm;
                const double err = ys_c[i] - (y0 + slope * u + 0.5 * kappa * u * u);
                se += err * err;
            }
            fit_mse = se / static_cast<double>(nc);
        }
    }

    const double conf = clamp_d(std::min(1.0, static_cast<double>(xs_c.size()) / 6.0) * std::exp(-3.0 * std::max(0.0, fit_mse)), 0.0, 1.0);
    build_corridor_estimate_c(inst->tracks, xm, y0, slope, kappa, conf, horizon_m, step_m, side_sign, *out_corridor);
    return 1;
}

namespace {

inline void compose_delta_rot_c(
    const double r_base[9],
    double d_roll,
    double d_pitch,
    double d_yaw,
    double r_out[9]
) {
    const double cr = std::cos(d_roll), sr = std::sin(d_roll);
    const double cp = std::cos(d_pitch), sp = std::sin(d_pitch);
    const double cy = std::cos(d_yaw), sy = std::sin(d_yaw);
    // R_delta = Rz(yaw) * Ry(pitch) * Rx(roll)
    const double rd[9] = {
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr
    };
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            r_out[r * 3 + c] = rd[r * 3 + 0] * r_base[0 + c] +
                               rd[r * 3 + 1] * r_base[3 + c] +
                               rd[r * 3 + 2] * r_base[6 + c];
        }
    }
}

inline bool solve_3x3_linear_c(const double a[9], const double b[3], double x[3]) {
    double m[3][4] = {
        {a[0], a[1], a[2], b[0]},
        {a[3], a[4], a[5], b[1]},
        {a[6], a[7], a[8], b[2]}
    };
    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double best = std::abs(m[col][col]);
        for (int r = col + 1; r < 3; ++r) {
            if (std::abs(m[r][col]) > best) {
                best = std::abs(m[r][col]);
                piv = r;
            }
        }
        if (best < 1e-12) return false;
        if (piv != col) {
            for (int c = col; c < 4; ++c) std::swap(m[col][c], m[piv][c]);
        }
        const double inv_p = 1.0 / m[col][col];
        for (int c = col; c < 4; ++c) m[col][c] *= inv_p;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = m[r][col];
            for (int c = col; c < 4; ++c) m[r][c] -= f * m[col][c];
        }
    }
    x[0] = m[0][3];
    x[1] = m[1][3];
    x[2] = m[2][3];
    return true;
}

}  // namespace

FUSION_CPP_EXPORT int32_t cpp_online_calibrate_extrinsics(
    int32_t n_pairs,
    const fusion_core_cpp::OnlineCalibPairC* pairs,
    const fusion_core_cpp::CameraIntrinsicsC* intr,
    const fusion_core_cpp::ExtrinsicsC* ext,
    int32_t min_pairs,
    fusion_core_cpp::OnlineCalibResultC* out_res
) {
    if (!pairs || !intr || !ext || !out_res || n_pairs <= 0) {
        return 0;
    }
    *out_res = {};
    for (int i = 0; i < 9; ++i) out_res->refined_r[i] = ext->r[i];
    for (int i = 0; i < 3; ++i) out_res->refined_t[i] = ext->t[i];

    auto eval_rmse = [&](const double r_cur[9]) -> std::pair<double, int32_t> {
        fusion_core_cpp::ExtrinsicsC tmp = *ext;
        for (int i = 0; i < 9; ++i) tmp.r[i] = r_cur[i];
        double se = 0.0;
        int32_t cnt = 0;
        for (int32_t i = 0; i < n_pairs; ++i) {
            double u = 0.0, v = 0.0, d = 0.0;
            if (project_lidar_point_c(pairs[i].cx, pairs[i].cy, pairs[i].cz, intr, &tmp, u, v, d)) {
                const double du = u - pairs[i].det_u;
                const double dv = v - pairs[i].det_v;
                se += du * du + dv * dv;
                ++cnt;
            }
        }
        return {(cnt > 0) ? std::sqrt(se / static_cast<double>(cnt)) : 1e6, cnt};
    };

    const auto [init_rmse, valid_cnt] = eval_rmse(ext->r);
    out_res->matched_pairs = valid_cnt;
    out_res->initial_rmse_px = init_rmse;
    out_res->final_rmse_px = init_rmse;
    if (valid_cnt < std::max<int32_t>(3, min_pairs)) {
        return 0;
    }

    double params[3] = {0.0, 0.0, 0.0};  // d_roll, d_pitch, d_yaw
    constexpr double kHuberDelta = 15.0;
    constexpr double kEps = 1e-4;
    for (int iter = 0; iter < 6; ++iter) {
        double r_cur[9];
        compose_delta_rot_c(ext->r, params[0], params[1], params[2], r_cur);
        fusion_core_cpp::ExtrinsicsC tmp = *ext;
        for (int i = 0; i < 9; ++i) tmp.r[i] = r_cur[i];

        double h_mat[9] = {1e-3, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3};
        double g_vec[3] = {0.0, 0.0, 0.0};
        for (int32_t i = 0; i < n_pairs; ++i) {
            double u0 = 0.0, v0 = 0.0, d0 = 0.0;
            if (!project_lidar_point_c(pairs[i].cx, pairs[i].cy, pairs[i].cz, intr, &tmp, u0, v0, d0)) {
                continue;
            }
            const double res[2] = {u0 - pairs[i].det_u, v0 - pairs[i].det_v};
            const double r_norm = std::hypot(res[0], res[1]);
            const double w = (r_norm <= kHuberDelta) ? 1.0 : (kHuberDelta / std::max(1e-6, r_norm));

            double j_col[3][2]{};
            for (int k = 0; k < 3; ++k) {
                double p_pert[3] = {params[0], params[1], params[2]};
                p_pert[k] += kEps;
                double r_p[9];
                compose_delta_rot_c(ext->r, p_pert[0], p_pert[1], p_pert[2], r_p);
                fusion_core_cpp::ExtrinsicsC tmp_p = *ext;
                for (int m = 0; m < 9; ++m) tmp_p.r[m] = r_p[m];
                double up = 0.0, vp = 0.0, dp = 0.0;
                if (project_lidar_point_c(pairs[i].cx, pairs[i].cy, pairs[i].cz, intr, &tmp_p, up, vp, dp)) {
                    j_col[k][0] = (up - u0) / kEps;
                    j_col[k][1] = (vp - v0) / kEps;
                }
            }
            for (int r = 0; r < 3; ++r) {
                g_vec[r] -= w * (j_col[r][0] * res[0] + j_col[r][1] * res[1]);
                for (int c = 0; c < 3; ++c) {
                    h_mat[r * 3 + c] += w * (j_col[r][0] * j_col[c][0] + j_col[r][1] * j_col[c][1]);
                }
            }
        }
        double step[3] = {0.0, 0.0, 0.0};
        if (!solve_3x3_linear_c(h_mat, g_vec, step)) break;
        params[0] = clamp_d(params[0] + clamp_d(step[0], -0.05, 0.05), -0.20, 0.20);
        params[1] = clamp_d(params[1] + clamp_d(step[1], -0.05, 0.05), -0.20, 0.20);
        params[2] = clamp_d(params[2] + clamp_d(step[2], -0.05, 0.05), -0.20, 0.20);
        if (std::abs(step[0]) < 1e-5 && std::abs(step[1]) < 1e-5 && std::abs(step[2]) < 1e-5) {
            break;
        }
    }

    double r_fin[9];
    compose_delta_rot_c(ext->r, params[0], params[1], params[2], r_fin);
    const auto [fin_rmse, fin_cnt] = eval_rmse(r_fin);
    if (fin_rmse <= init_rmse + 1e-6 && fin_cnt >= std::max<int32_t>(3, min_pairs)) {
        out_res->final_rmse_px = fin_rmse;
        out_res->delta_roll_rad = params[0];
        out_res->delta_pitch_rad = params[1];
        out_res->delta_yaw_rad = params[2];
        for (int i = 0; i < 9; ++i) out_res->refined_r[i] = r_fin[i];
        return 1;
    }
    return 0;
}
