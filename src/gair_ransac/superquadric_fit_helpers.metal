inline float sq_logaddexp(float a, float b) {
    float m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

template <bool with_jacobian>
inline float sq_radial(
    float3 point,
    thread const float* p,
    float radial_eps,
    thread float* jac
) {
    float cy = cos(p[5]), sy = sin(p[5]);
    float cp = cos(p[6]), sp = sin(p[6]);
    float cr = cos(p[7]), sr = sin(p[7]);
    float3x3 rz(float3(cy, sy, 0), float3(-sy, cy, 0), float3(0, 0, 1));
    float3x3 ry(float3(cp, 0, -sp), float3(0, 1, 0), float3(sp, 0, cp));
    float3x3 rx(float3(1, 0, 0), float3(0, cr, sr), float3(0, -sr, cr));
    float3x3 rotation = rz * ry * rx;
    float3 centered = point - float3(p[8], p[9], p[10]);
    float3 pc = transpose(rotation) * centered;
    float3 axes(p[0], p[1], p[2]);
    float3 q = pc / axes;
    float3 absq = abs(q);
    float3 safeq = max(absq, float3(1e-12f));
    float3 logq = log(safeq);
    float pxy = 2.0f / p[4], pz = 2.0f / p[3];
    float k = p[4] / p[3], h = -0.5f * p[3];
    float loga = pxy * logq.x, logb = pxy * logq.y;
    float logu_raw = sq_logaddexp(loga, logb);
    float logu = max(logu_raw, log(1e-12f));
    float logv = k * logu, logw = pz * logq.z;
    float logs_raw = sq_logaddexp(logv, logw);
    float logs = max(logs_raw, log(1e-12f));
    float r_raw = length(pc), r = max(r_raw, radial_eps);
    float sh = exp(h * logs);
    float residual = r * (1.0f - sh);
    if (!with_jacobian) {
        return residual;
    }

    float ua = logu_raw > log(1e-12f) ? exp(loga - logu) : 0.0f;
    float ub = logu_raw > log(1e-12f) ? exp(logb - logu) : 0.0f;
    float sv = logs_raw > log(1e-12f) ? exp(logv - logs) : 0.0f;
    float sw = logs_raw > log(1e-12f) ? exp(logw - logs) : 0.0f;
    float3 active = select(float3(0), float3(1), absq > float3(1e-12f));
    float3 dlogs_dpc = float3(sv * k * ua * pxy, sv * k * ub * pxy, sw * pz)
        * active * sign(q) / (safeq * axes);
    float3 grad_r = r_raw > radial_eps ? pc / r_raw : float3(0);
    float3 grad_pc = (1.0f - sh) * grad_r - r * sh * h * dlogs_dpc;

    jac[0] = r * sh * h * sv * k * ua * pxy * active.x / axes.x;
    jac[1] = r * sh * h * sv * k * ub * pxy * active.y / axes.y;
    jac[2] = r * sh * h * sw * pz * active.z / axes.z;
    float dlogs_de1 = sv * (-k / p[3]) * logu + sw * (-pz / p[3]) * logq.z;
    jac[3] = -r * sh * (-0.5f * logs + h * dlogs_de1);
    float dlogu_de2 = (-pxy / p[4]) * (ua * logq.x + ub * logq.y);
    float dlogs_de2 = sv * (logu / p[3] + k * dlogu_de2);
    jac[4] = -r * sh * h * dlogs_de2;

    float3x3 drz(float3(-sy, cy, 0), float3(-cy, -sy, 0), float3(0));
    float3x3 dry(float3(-sp, 0, -cp), float3(0), float3(cp, 0, -sp));
    float3x3 drx(float3(0), float3(0, -sr, cr), float3(0, -cr, -sr));
    jac[5] = dot(grad_pc, transpose(drz * ry * rx) * centered);
    jac[6] = dot(grad_pc, transpose(rz * dry * rx) * centered);
    jac[7] = dot(grad_pc, transpose(rz * ry * drx) * centered);
    float3 grad_translation = -(rotation * grad_pc);
    jac[8] = grad_translation.x;
    jac[9] = grad_translation.y;
    jac[10] = grad_translation.z;
    return residual;
}

inline float sq_soft_l1(float residual, float loss_scale) {
    float ratio = residual / loss_scale;
    return residual * residual / (sqrt(1.0f + ratio * ratio) + 1.0f);
}

template <bool with_jacobian>
inline float sq_axis_penalty(
    int axis,
    thread const float* p,
    thread const float* box,
    float residual_scale,
    thread float* jac
) {
    float cy = cos(p[5]), sy = sin(p[5]);
    float cp = cos(p[6]), sp = sin(p[6]);
    float cr = cos(p[7]), sr = sin(p[7]);
    float3x3 rz(float3(cy, sy, 0), float3(-sy, cy, 0), float3(0, 0, 1));
    float3x3 ry(float3(cp, 0, -sp), float3(0, 1, 0), float3(sp, 0, cp));
    float3x3 rx(float3(1, 0, 0), float3(0, cr, sr), float3(0, -sr, cr));
    float3x3 rotation = rz * ry * rx;
    float3x3 support_box(float3(box[0], box[3], box[6]),
                         float3(box[1], box[4], box[7]),
                         float3(box[2], box[5], box[8]));
    float3 projection = transpose(support_box) * rotation[axis];
    float support = dot(abs(projection), float3(1));
    float excess = max(p[axis] / support - 1.0f, 0.0f);
    if (with_jacobian) {
        for (int j = 0; j < 11; ++j) jac[j] = 0.0f;
        if (excess > 0.0f) {
            jac[axis] = residual_scale / support;
            float3x3 drz(float3(-sy, cy, 0), float3(-cy, -sy, 0), float3(0));
            float3x3 dry(float3(-sp, 0, -cp), float3(0), float3(cp, 0, -sp));
            float3x3 drx(float3(0), float3(0, -sr, cr), float3(0, -cr, -sr));
            float factor = -residual_scale * p[axis] / (support * support);
            jac[5] = factor * dot(sign(projection), transpose(support_box) * (drz * ry * rx)[axis]);
            jac[6] = factor * dot(sign(projection), transpose(support_box) * (rz * dry * rx)[axis]);
            jac[7] = factor * dot(sign(projection), transpose(support_box) * (rz * ry * drx)[axis]);
        }
    }
    return residual_scale * excess;
}
