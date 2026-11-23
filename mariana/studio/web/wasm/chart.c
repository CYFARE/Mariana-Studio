/*
 * Loss-chart decimation for the GUI.
 *
 * A long run can produce far more points than the chart has pixels.  The
 * renderer keeps the minimum and maximum value in every pixel column, which
 * preserves spikes and keeps the polyline small.  This kernel is compiled once
 * and shipped as chart.wasm; the browser uses it when WebAssembly is available
 * and falls back to the same algorithm in JavaScript otherwise.
 *
 * Regenerate from this directory with:
 *   clang --target=wasm32 -O3 -nostdlib -fno-builtin \
 *     -Wl,--no-entry -Wl,--export=decimate -Wl,--export=steps_in \
 *     -Wl,--export=values_in -Wl,--export=steps_out -Wl,--export=values_out \
 *     -Wl,--export-memory -Wl,--initial-memory=12582912 \
 *     -o chart.wasm chart.c
 */

#define MAX_POINTS 200000

static double input_steps[MAX_POINTS];
static double input_values[MAX_POINTS];
static double output_steps[MAX_POINTS * 2];
static double output_values[MAX_POINTS * 2];

double *steps_in(void) { return input_steps; }
double *values_in(void) { return input_values; }
double *steps_out(void) { return output_steps; }
double *values_out(void) { return output_values; }

static int emit_bin(int bin, int min_index, int max_index, int out) {
    if (min_index == max_index) {
        output_steps[out] = input_steps[min_index];
        output_values[out] = input_values[min_index];
        return out + 1;
    }
    int first = min_index < max_index ? min_index : max_index;
    int second = first == min_index ? max_index : min_index;
    output_steps[out] = input_steps[first];
    output_values[out] = input_values[first];
    output_steps[out + 1] = input_steps[second];
    output_values[out + 1] = input_values[second];
    return out + 2;
}

int decimate(int count, double min_step, double max_step, int bins) {
    if (count <= 0 || count > MAX_POINTS || bins <= 0 || max_step <= min_step) {
        return 0;
    }
    int out = 0;
    int bin = -1;
    int min_index = 0;
    int max_index = 0;
    double bin_min = 0.0;
    double bin_max = 0.0;
    double span = max_step - min_step;
    for (int index = 0; index < count; index++) {
        double step = input_steps[index];
        double value = input_values[index];
        if (value != value) continue;  /* NaN */
        int next = (int)(((step - min_step) / span) * bins);
        if (next < 0) next = 0;
        if (next >= bins) next = bins - 1;
        if (next != bin) {
            if (bin >= 0) out = emit_bin(bin, min_index, max_index, out);
            bin = next;
            min_index = max_index = index;
            bin_min = bin_max = value;
        } else {
            if (value < bin_min) {
                bin_min = value;
                min_index = index;
            }
            if (value > bin_max) {
                bin_max = value;
                max_index = index;
            }
        }
    }
    if (bin >= 0) out = emit_bin(bin, min_index, max_index, out);
    return out;
}
