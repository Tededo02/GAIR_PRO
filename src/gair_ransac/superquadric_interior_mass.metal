uint lane = thread_position_in_threadgroup.x;
uint candidate = threadgroup_position_in_grid.y;
uint point_index = thread_position_in_grid.x;
uint point_count = uint(interior_shape[1]);
uint neighbor_count = uint(neighbors_shape[1]);
uint tile = threadgroup_position_in_grid.x;
uint tiles = (point_count + THREADS - 1) / THREADS;
threadgroup float sums[THREADS / 32];
float contribution = 0.0f;
if (point_index < point_count) {
    uint offset = candidate * point_count;
    float strength = interior[offset + point_index];
    if (strength > 0.0f) {
        float neighbor_sum = 0.0f;
        uint valid = 0;
        for (uint j = 0; j < neighbor_count; ++j) {
            uint neighbor = neighbors[point_index * neighbor_count + j];
            if (neighbor < point_count) {
                neighbor_sum += interior[offset + neighbor];
                valid += 1;
            }
        }
        contribution = strength * (valid ? neighbor_sum / float(valid) : 1.0f);
    }
}
float reduced = simd_sum(contribution);
if (lane % 32 == 0) sums[lane / 32] = reduced;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    float total = 0.0f;
    for (uint group = 0; group < THREADS / 32; ++group) total += sums[group];
    mass_partials[candidate * tiles + tile] = total;
}
