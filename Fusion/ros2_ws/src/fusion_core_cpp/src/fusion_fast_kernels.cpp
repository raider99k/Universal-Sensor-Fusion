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
                          (cnt <= max_pts);
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
        if (h_val < eff_min_h || h_val > c_max_h || rad > c_max_r) {
            continue;
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
