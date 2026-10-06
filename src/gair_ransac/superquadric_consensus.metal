uint lane = thread_position_in_threadgroup.x;
uint candidate = threadgroup_position_in_grid.y;
uint point_index = thread_position_in_grid.x;
uint point_count = uint(points_shape[0]);
uint tile = threadgroup_position_in_grid.x;
uint tiles = (point_count + THREADS - 1) / THREADS;
threadgroup uint2 sums[THREADS / 32];
uint inlier = 0, uncertain = 0;

if (point_index < point_count) {
    uint base = 17 * candidate;
    float3 axes(parameters[base], parameters[base+1], parameters[base+2]);
    float e1 = parameters[base+3], e2 = parameters[base+4];
    float3 displacement(
        points[3*point_index] - parameters[base+5],
        points[3*point_index+1] - parameters[base+6],
        points[3*point_index+2] - parameters[base+7]
    );
    float3 r0(parameters[base+8], parameters[base+9], parameters[base+10]);
    float3 r1(parameters[base+11], parameters[base+12], parameters[base+13]);
    float3 r2(parameters[base+14], parameters[base+15], parameters[base+16]);
    float3 pc = displacement.x*r0 + displacement.y*r1 + displacement.z*r2;
    float3 ratios = abs(pc / axes);
    float pxy = 2.0f / e2, pz = 2.0f / e1, k = e2 / e1;
    float log_u = sq_logadd(pxy*log(ratios.x), pxy*log(ratios.y));
    float raw_log_shape = sq_logadd(k*log_u, pz*log(ratios.z));
    float log_shape = max(raw_log_shape, log(1e-12f));
    float raw_radius = length(pc);
    float radius = max(raw_radius, options[2]);
    float surface_radius = radius * exp(-0.5f * e1 * log_shape);
    float error = abs(radius - surface_radius);
    float threshold = options[0];
    bool is_active = true;
    if (SCORE_INTERIOR) {
        is_active = active[point_index] != 0;
        float depth = 0.0f;
        if (raw_radius <= options[2]) {
            depth = min(axes.x, min(axes.y, axes.z)) * exp2(-0.5f*(max(e1-1.0f, 0.0f)+max(e2-1.0f, 0.0f)));
        } else if (raw_log_shape < 0.0f) {
            depth = raw_log_shape == log_shape ? surface_radius - raw_radius
                : exp(log(raw_radius) - 0.5f*e1*raw_log_shape) - raw_radius;
        }
        float excess = max(depth-threshold, 0.0f);
        float denominator = max(excess, threshold);
        float x = excess / denominator, y = threshold / denominator;
        interior[candidate*point_count+point_index] = x*x / (x*x+y*y);
    }
    float distance_margin = 128.0f * FLT_EPSILON * max(1.0f, radius + surface_radius);
    uncertain = is_active && isfinite(threshold) && abs(error-threshold) <= distance_margin;
    bool accepted = is_active && error < threshold;

    if (HAS_NORMALS && accepted && !uncertain) {
        float3 observed(normals[3*point_index], normals[3*point_index+1], normals[3*point_index+2]);
        if (dot(observed, observed) == 0.0f) {
            accepted = 0.0f >= options[1];
        } else {
            float3 log_ratios = log(max(ratios, float3(1e-12f)));
            float normal_log_u = max(sq_logadd(pxy*log_ratios.x, pxy*log_ratios.y), log(1e-12f));
            float3 log_gradient(
                log(k*pxy/axes.x) - options[3] + (k-1.0f)*normal_log_u + (pxy-1.0f)*log_ratios.x,
                log(k*pxy/axes.y) - options[3] + (k-1.0f)*normal_log_u + (pxy-1.0f)*log_ratios.y,
                log(pz/axes.z) - options[3] + (pz-1.0f)*log_ratios.z
            );
            log_gradient = select(log_gradient, float3(-INFINITY), pc == 0.0f);
            float largest = max(log_gradient.x, max(log_gradient.y, log_gradient.z));
            float3 gradient = largest == -INFINITY ? float3(0.0f) : exp(log_gradient-largest) * sign(pc);
            float3 world_gradient(dot(gradient, r0), dot(gradient, r1), dot(gradient, r2));
            float gradient_length = length(world_gradient);
            float amplitude = gradient_length == 0.0f ? 0.0f : min(1.0f, exp(min(largest + log(gradient_length) - log(1e-9f), 0.0f)));
            float3 model_normal = gradient_length == 0.0f ? float3(0.0f) : amplitude * world_gradient / gradient_length;
            float alignment = clamp(dot(model_normal, observed), -1.0f, 1.0f);
            uncertain = abs(alignment-options[1]) <= 128.0f*FLT_EPSILON
                || min(ratios.x, min(ratios.y, ratios.z)) < 128.0f*FLT_EPSILON
                || largest > 350.0f || !isfinite(alignment);
            accepted = alignment >= options[1];
        }
    }
    inlier = accepted && !uncertain;
    masks[candidate*point_count+point_index] = uchar(uncertain ? 2 : inlier);
}

uint2 reduced(simd_sum(inlier), simd_sum(uncertain));
if (lane % 32 == 0) sums[lane / 32] = reduced;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    uint2 total(0);
    for (uint group = 0; group < THREADS/32; ++group) total += sums[group];
    uint offset = 2*(candidate*tiles+tile);
    partials[offset] = total.x;
    partials[offset+1] = total.y;
}
