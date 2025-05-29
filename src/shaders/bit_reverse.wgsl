// Bit-reverse reordering compute shader
// This reorders the input data according to bit-reversed indices
// Required for Cooley-Tukey FFT algorithm

struct Complex {
    real: f32,
    imag: f32,
}

struct FftParams {
    size: u32,
    inverse: u32,
    stage: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<Complex>;
@group(0) @binding(1) var<uniform> params: FftParams;

// Bit-reverse a number with given number of bits
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var result = 0u;
    var value = x;
    for (var i = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | (value & 1u);
        value = value >> 1u;
    }
    return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let size = params.size;
    
    if (thread_id >= size) {
        return;
    }
    
    // Calculate log2 of size
    var log2_size = 0u;
    var temp_size = size;
    while (temp_size > 1u) {
        temp_size = temp_size >> 1u;
        log2_size = log2_size + 1u;
    }
    
    // Calculate bit-reversed index
    let reversed_idx = bit_reverse(thread_id, log2_size);
    
    // Only swap if the reversed index is greater than the current index
    // This ensures each pair is swapped exactly once
    if (reversed_idx > thread_id) {
        let temp = data[thread_id];
        data[thread_id] = data[reversed_idx];
        data[reversed_idx] = temp;
    }
} 