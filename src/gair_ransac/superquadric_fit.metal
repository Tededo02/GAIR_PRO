uint lane = thread_position_in_threadgroup.x;
uint fit_index = threadgroup_position_in_grid.y;
uint simd_group = lane / 32, simd_lane = lane % 32;
uint point_count = uint(points_shape[1]);
uint point_offset = fit_index * point_count * 3;
threadgroup float parameters[11], candidate[11], gradient[11], hessian[121];
threadgroup float lower[11], upper[11], step[11], scales[11];
threadgroup float partials[THREADS / 32][78];
threadgroup float current_cost, damping, growth, predicted, step_norm;
threadgroup int status, evaluations, accepted;
float count = float(point_count);
float loss_scale = options[3 * fit_index], radial_eps = options[3 * fit_index + 1];
int max_evaluations = int(options[3 * fit_index + 2]);
if (lane < 11) {
    parameters[lane] = initial[11 * fit_index + lane];
    lower[lane] = bounds[22 * fit_index + lane];
    upper[lane] = bounds[22 * fit_index + 11 + lane];
}
if (lane == 0) {
    damping = 1e-3f;
    growth = 2.0f;
    status = 0;
    evaluations = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

while (evaluations < max_evaluations && status == 0) {
    float local_p[11], local_g[11] = {}, local_h[66] = {};
    for (int j = 0; j < 11; ++j) local_p[j] = parameters[j];
    float local_cost = 0.0f;
    for (uint i = lane; i < point_count; i += THREADS) {
        float jac[11];
        float3 point(points[point_offset+3*i], points[point_offset+3*i+1], points[point_offset+3*i+2]);
        float residual = sq_radial<true>(point, local_p, radial_eps, jac);
        float ratio = residual / loss_scale;
        float weight = rsqrt(1.0f + ratio * ratio);
        local_cost += sq_soft_l1(residual, loss_scale);
        int index = 0;
        for (int j = 0; j < 11; ++j) {
            local_g[j] += jac[j] * residual * weight;
            for (int k = 0; k <= j; ++k) {
                local_h[index++] += jac[j] * jac[k] * weight;
            }
        }
    }
    float cost = simd_sum(local_cost);
    if (simd_lane == 0) partials[simd_group][0] = cost;
    for (int j = 0; j < 11; ++j) {
        float value = simd_sum(local_g[j]);
        if (simd_lane == 0) partials[simd_group][1+j] = value;
    }
    for (int index = 0; index < 66; ++index) {
        float value = simd_sum(local_h[index]);
        if (simd_lane == 0) partials[simd_group][12+index] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        current_cost = 0.0f;
        for (int group = 0; group < THREADS / 32; ++group) current_cost += partials[group][0] / count;
        for (int j = 0; j < 11; ++j) {
            gradient[j] = 0.0f;
            for (int group = 0; group < THREADS / 32; ++group) gradient[j] += partials[group][1+j] / count;
        }
        int index = 0;
        for (int j = 0; j < 11; ++j) {
            for (int k = 0; k <= j; ++k) {
                float value = 0.0f;
                for (int group = 0; group < THREADS / 32; ++group) value += partials[group][12+index] / count;
                hessian[11*j+k] = value;
                hessian[11*k+j] = value;
                index += 1;
            }
        }
        evaluations += 1;
        float projected_gradient = 0.0f;
        for (int j = 0; j < 11; ++j) {
            scales[j] = sqrt(max(hessian[11*j+j], 1e-12f));
            float distance = gradient[j] > 0.0f ? parameters[j] - lower[j] : upper[j] - parameters[j];
            projected_gradient = max(projected_gradient, abs(gradient[j]) * min(distance, 1.0f));
            if (!isfinite(gradient[j]) || !isfinite(scales[j])) status = -1;
        }
        if (!isfinite(current_cost)) status = -1;
        if (status == 0 && projected_gradient < 1e-8f) status = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) accepted = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while (!accepted && evaluations < max_evaluations && status == 0) {
        if (lane == 0) {
            float factor[121] = {}, rhs[11], delta[11];
            bool blocked[11];
            for (int j = 0; j < 11; ++j) {
                blocked[j] = (parameters[j] <= lower[j] && gradient[j] > 0.0f)
                    || (parameters[j] >= upper[j] && gradient[j] < 0.0f);
                rhs[j] = blocked[j] ? 0.0f : -gradient[j] / scales[j];
            }
            bool valid = true;
            for (int j = 0; j < 11; ++j) {
                for (int k = 0; k <= j; ++k) {
                    float value = (blocked[j] || blocked[k]) ? 0.0f
                        : hessian[11*j+k] / (scales[j] * scales[k]);
                    if (j == k) value += blocked[j] ? 1.0f : damping;
                    for (int m = 0; m < k; ++m) value -= factor[11*j+m] * factor[11*k+m];
                    if (j == k) {
                        valid = valid && value > 0.0f && isfinite(value);
                        factor[11*j+k] = sqrt(max(value, 1e-20f));
                    } else {
                        factor[11*j+k] = value / factor[11*k+k];
                    }
                }
            }
            for (int j = 0; j < 11; ++j) {
                float value = rhs[j];
                for (int k = 0; k < j; ++k) value -= factor[11*j+k] * delta[k];
                delta[j] = value / factor[11*j+j];
            }
            for (int j = 10; j >= 0; --j) {
                float value = delta[j];
                for (int k = j + 1; k < 11; ++k) value -= factor[11*k+j] * delta[k];
                delta[j] = value / factor[11*j+j];
            }
            step_norm = 0.0f;
            for (int j = 0; j < 11; ++j) {
                candidate[j] = clamp(parameters[j] + delta[j] / scales[j], lower[j], upper[j]);
                step[j] = candidate[j] - parameters[j];
                step_norm = max(step_norm, abs(step[j]) / (1.0f + abs(parameters[j])));
            }
            predicted = 0.0f;
            for (int j = 0; j < 11; ++j) {
                predicted -= gradient[j] * step[j];
                for (int k = 0; k < 11; ++k) predicted -= 0.5f * step[j] * hessian[11*j+k] * step[k];
            }
            if (!valid || !isfinite(predicted)) status = -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < 11; ++j) local_p[j] = candidate[j];
        local_cost = 0.0f;
        for (uint i = lane; i < point_count; i += THREADS) {
            float unused[11];
            float3 point(points[point_offset+3*i], points[point_offset+3*i+1], points[point_offset+3*i+2]);
            float residual = sq_radial<false>(point, local_p, radial_eps, unused);
            local_cost += sq_soft_l1(residual, loss_scale);
        }
        float partial_cost = simd_sum(local_cost);
        if (simd_lane == 0) partials[simd_group][0] = partial_cost;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && status == 0) {
            float trial_cost = 0.0f;
            for (int group = 0; group < THREADS / 32; ++group) trial_cost += partials[group][0] / count;
            evaluations += 1;
            float reduction = current_cost - trial_cost;
            float gain = predicted > 0.0f ? reduction / predicted : -1.0f;
            if (isfinite(trial_cost) && reduction > 0.0f && gain > 0.0f) {
                accepted = 1;
                for (int j = 0; j < 11; ++j) parameters[j] = candidate[j];
                damping = max(1e-9f, damping * max(1.0f / 3.0f, 1.0f - pow(2.0f * gain - 1.0f, 3.0f)));
                growth = 2.0f;
                if (reduction < 1e-6f * current_cost && gain > 0.25f) status = 2;
                if (step_norm < 1e-7f) status = 3;
                current_cost = trial_cost;
            } else {
                damping *= growth;
                growth = min(growth * 2.0f, 1e6f);
                if (step_norm < 1e-7f) status = 3;
                if (!isfinite(damping)) status = -1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
if (lane < 11) fitted[11 * fit_index + lane] = parameters[lane];
if (lane == 0) {
    diagnostics[3 * fit_index] = float(status);
    diagnostics[3 * fit_index + 1] = float(evaluations);
    diagnostics[3 * fit_index + 2] = current_cost;
}
