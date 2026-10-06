#define SF_SMALL_BEND 1e-3f

template <bool spatial_derivatives, bool parameter_derivatives>
inline float3 sf_inverse_bend(
    float3 point, float2 components, int axis,
    thread float3x3& spatial, thread float3* parameter
) {
    int order[3] = {(axis + 1) % 3, (axis + 2) % 3, axis};
    float3 local(point[order[0]], point[order[1]], point[order[2]]);
    float2 xy = local.xy;
    float z = local.z, curvature = length(components);
    float3 result = local;
    float3x3 step = float3x3(1.0f);
    float3 derivative[2] = {float3(0), float3(0)};
    float extent = max(1.0f, max(abs(local.x), max(abs(local.y), abs(local.z))));
    if (curvature * extent < SF_SMALL_BEND) {
        float projection = dot(xy, components), k2 = dot(components, components);
        result.xy -= 0.5f * z*z * components * (1.0f + projection);
        result.z = z * (1.0f + projection + projection*projection) - k2*z*z*z / 3.0f;
        if (spatial_derivatives) {
            for (int col = 0; col < 2; ++col) {
                for (int row = 0; row < 2; ++row) step[col][row] -= 0.5f*z*z*components[row]*components[col];
                step[col].z = z * (1.0f + 2.0f*projection) * components[col];
            }
            step[2] = float3(-z*components*(1.0f + projection), 1.0f + projection + projection*projection - k2*z*z);
        }
        if (parameter_derivatives) {
            for (int col = 0; col < 2; ++col) {
                for (int row = 0; row < 2; ++row) {
                    derivative[col][row] = -0.5f*z*z * ((row == col ? 1.0f + projection : 0.0f) + components[row]*xy[col]);
                }
                derivative[col].z = z*(1.0f + 2.0f*projection)*xy[col] - (2.0f/3.0f)*z*z*z*components[col];
            }
        }
    } else {
        float2 direction = components / curvature;
        float2 perpendicular(-direction.y, direction.x);
        float radial = dot(xy, direction), h = 1.0f - curvature*radial;
        float magnitude = max(length(float2(h, curvature*z)), 1e-12f);
        float numerator = 2.0f*radial - curvature*(radial*radial + z*z);
        float unbent_radial = numerator / (1.0f + magnitude);
        float shift = unbent_radial - radial, angle = atan2(curvature*z, h);
        result.xy += shift * direction;
        result.z = angle / curvature;
        if (spatial_derivatives || parameter_derivatives) {
            float shift_r = h/magnitude - 1.0f, shift_z = -curvature*z/magnitude;
            float longitudinal_r = curvature*z/(magnitude*magnitude), longitudinal_z = h/(magnitude*magnitude);
            if (spatial_derivatives) {
                for (int col = 0; col < 2; ++col) {
                    for (int row = 0; row < 2; ++row) step[col][row] += shift_r*direction[row]*direction[col];
                    step[col].z = longitudinal_r * direction[col];
                }
                step[2] = float3(shift_z*direction, longitudinal_z);
            }
            if (parameter_derivatives) {
                float magnitude_k = (-radial*h + curvature*z*z)/magnitude;
                float shift_k = -(radial*radial + z*z)/(1.0f + magnitude) - numerator*magnitude_k/((1.0f + magnitude)*(1.0f + magnitude));
                float longitudinal_k = (curvature*z/(magnitude*magnitude) - angle)/(curvature*curvature);
                float radial_alpha = dot(xy, perpendicular);
                float3 alpha_derivative(shift_r*radial_alpha*direction + shift*perpendicular, longitudinal_r*radial_alpha);
                float3 k_derivative(shift_k*direction, longitudinal_k);
                for (int col = 0; col < 2; ++col) derivative[col] = k_derivative*direction[col] + alpha_derivative*perpendicular[col]/curvature;
            }
        }
    }
    float3 global;
    for (int row = 0; row < 3; ++row) global[order[row]] = result[row];
    if (spatial_derivatives) {
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) spatial[order[col]][order[row]] = step[col][row];
        }
    }
    if (parameter_derivatives) {
        for (int col = 0; col < 2; ++col) {
            for (int row = 0; row < 3; ++row) parameter[col][order[row]] = derivative[col][row];
        }
    }
    return global;
}

template <bool spatial_derivatives, bool parameter_derivatives>
inline float3 sf_inverse_deformation(
    float3 point, thread const float* p,
    thread float3x3& spatial, thread float3* parameter
) {
    if (spatial_derivatives) spatial = float3x3(1.0f);
    if (parameter_derivatives) for (int j = 0; j < 9; ++j) parameter[j] = float3(0);
    int sequence[3] = {2, 0, 1};
    for (int stage = 0; stage < 3; ++stage) {
        int axis = sequence[stage];
        float3x3 step;
        float3 derivative[2];
        point = sf_inverse_bend<spatial_derivatives, parameter_derivatives>(
            point, float2(p[13+2*axis], p[14+2*axis]), axis, step, derivative);
        if (spatial_derivatives) spatial = step * spatial;
        if (parameter_derivatives) {
            for (int j = 0; j < 9; ++j) parameter[j] = step * parameter[j];
            parameter[3+2*axis] += derivative[0];
            parameter[4+2*axis] += derivative[1];
        }
    }
    float2 taper(p[11], p[12]);
    float2 raw_denominator = 1.0f + point.z*taper/p[2];
    float2 denominator = max(raw_denominator, float2(1e-6f));
    if (spatial_derivatives) {
        float2 active = select(float2(0), float2(1), raw_denominator > 1e-6f);
        float3x3 step(float3(1.0f/denominator.x, 0, 0), float3(0, 1.0f/denominator.y, 0),
            float3(-active*point.xy*taper/p[2]/(denominator*denominator), 1));
        spatial = step * spatial;
        if (parameter_derivatives) {
            for (int j = 0; j < 9; ++j) parameter[j] = step * parameter[j];
            parameter[0].xy += active*point.xy*taper*point.z/(p[2]*p[2]*denominator*denominator);
            parameter[1].x -= active.x*point.x*point.z/p[2]/(denominator.x*denominator.x);
            parameter[2].y -= active.y*point.y*point.z/p[2]/(denominator.y*denominator.y);
        }
    }
    return float3(point.xy/denominator, point.z);
}

template <bool with_jacobian>
inline float sf_radial(float3 point, thread const float* p, float radial_eps, thread float* jac) {
    float cy = cos(p[5]), sy = sin(p[5]), cp = cos(p[6]), sp = sin(p[6]), cr = cos(p[7]), sr = sin(p[7]);
    float3x3 rz(float3(cy, sy, 0), float3(-sy, cy, 0), float3(0, 0, 1));
    float3x3 ry(float3(cp, 0, -sp), float3(0, 1, 0), float3(sp, 0, cp));
    float3x3 rx(float3(1, 0, 0), float3(0, cr, sr), float3(0, -sr, cr));
    float3x3 rotation = rz * ry * rx;
    float3 centered = point - float3(p[8], p[9], p[10]);
    float3 local = transpose(rotation) * centered;
    float3x3 inverse_jacobian;
    float3 deformation_derivatives[9];
    float3 canonical = sf_inverse_deformation<with_jacobian, with_jacobian>(local, p, inverse_jacobian, deformation_derivatives);
    float3 axes(p[0], p[1], p[2]), q = canonical / axes, absq = abs(q);
    float3 safeq = max(absq, float3(1e-12f)), logq = log(safeq);
    float pxy = 2.0f/p[4], pz = 2.0f/p[3], k = p[4]/p[3], h = -0.5f*p[3];
    float loga = pxy*logq.x, logb = pxy*logq.y;
    float logu_raw = sq_logaddexp(loga, logb), logu = max(logu_raw, log(1e-12f));
    float logv = k*logu, logw = pz*logq.z;
    float logs_raw = sq_logaddexp(logv, logw), logs = max(logs_raw, log(1e-12f));
    float r_raw = length(local), r = max(r_raw, radial_eps), sh = exp(h*logs);
    float residual = r*(1.0f - sh);
    if (!with_jacobian) return residual;
    float ua = logu_raw > log(1e-12f) ? exp(loga-logu) : 0.0f;
    float ub = logu_raw > log(1e-12f) ? exp(logb-logu) : 0.0f;
    float sv = logs_raw > log(1e-12f) ? exp(logv-logs) : 0.0f;
    float sw = logs_raw > log(1e-12f) ? exp(logw-logs) : 0.0f;
    float3 active = select(float3(0), float3(1), absq > 1e-12f);
    float3 dlogs = float3(sv*k*ua*pxy, sv*k*ub*pxy, sw*pz) * active * sign(q) / (safeq*axes);
    float3 canonical_gradient = -r*sh*h*dlogs;
    jac[0] = r*sh*h*sv*k*ua*pxy*active.x/axes.x;
    jac[1] = r*sh*h*sv*k*ub*pxy*active.y/axes.y;
    jac[2] = r*sh*h*sw*pz*active.z/axes.z + dot(canonical_gradient, deformation_derivatives[0]);
    float dlogs_de1 = sv*(-k/p[3])*logu + sw*(-pz/p[3])*logq.z;
    jac[3] = -r*sh*(-0.5f*logs + h*dlogs_de1);
    float dlogu_de2 = (-pxy/p[4])*(ua*logq.x + ub*logq.y);
    jac[4] = -r*sh*h*sv*(logu/p[3] + k*dlogu_de2);
    float3 grad_local = transpose(inverse_jacobian)*canonical_gradient + (r_raw > radial_eps ? (1.0f-sh)*local/r_raw : float3(0));
    float3x3 drz(float3(-sy, cy, 0), float3(-cy, -sy, 0), float3(0));
    float3x3 dry(float3(-sp, 0, -cp), float3(0), float3(cp, 0, -sp));
    float3x3 drx(float3(0), float3(0, -sr, cr), float3(0, -cr, -sr));
    jac[5] = dot(grad_local, transpose(drz*ry*rx)*centered);
    jac[6] = dot(grad_local, transpose(rz*dry*rx)*centered);
    jac[7] = dot(grad_local, transpose(rz*ry*drx)*centered);
    float3 translation_gradient = -(rotation*grad_local);
    jac[8] = translation_gradient.x; jac[9] = translation_gradient.y; jac[10] = translation_gradient.z;
    for (int j = 0; j < 8; ++j) jac[11+j] = dot(canonical_gradient, deformation_derivatives[1+j]);
    return residual;
}

template <int PARAMETERS, bool with_jacobian>
inline float sq_model_radial(float3 point, thread const float* p, float radial_eps, thread float* jac) {
    if (PARAMETERS == 19) return sf_radial<with_jacobian>(point, p, radial_eps, jac);
    return sq_radial<with_jacobian>(point, p, radial_eps, jac);
}

template <int PARAMETERS, bool with_jacobian>
inline float sq_model_axis_penalty(int axis, thread const float* p, thread const float* box, float scale, thread float* jac) {
    float residual = sq_axis_penalty<with_jacobian>(axis, p, box, scale, jac);
    if (with_jacobian) for (int j = 11; j < PARAMETERS; ++j) jac[j] = 0.0f;
    return residual;
}
