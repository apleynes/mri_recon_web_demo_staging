// FFT Compute Shader - Radix-4 Cooley-Tukey Algorithm
// Based on RustFFT implementation

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

// Complex multiplication
fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

// Complex addition
fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

// Complex subtraction
fn complex_sub(a: Complex, b: Complex) -> Complex {
    return Complex(a.real - b.real, a.imag - b.imag);
}

// Generate twiddle factor for FFT
fn twiddle_factor(k: u32, n: u32, inverse: bool) -> Complex {
    let angle = -2.0 * 3.14159265359 * f32(k) / f32(n);
    let actual_angle = select(angle, -angle, inverse);
    return Complex(cos(actual_angle), sin(actual_angle));
}

// Cooley-Tukey FFT stage
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let size = params.size;
    let stage = params.stage;
    let inverse = params.inverse != 0u;
    
    if (thread_id >= size) {
        return;
    }
    
    let stage_size = 1u << (stage + 1u);
    let half_stage = 1u << stage;
    
    // Calculate which pair this thread processes
    let group_id = thread_id / half_stage;
    let pair_id = thread_id % half_stage;
    
    if (group_id % 2u != 0u) {
        return; // Only process even groups to avoid duplication
    }
    
    let base_idx = group_id * stage_size + pair_id;
    let pair_idx = base_idx + half_stage;
    
    if (pair_idx >= size) {
        return;
    }
    
    // Load the pair of values
    let a = data[base_idx];
    let b = data[pair_idx];
    
    // Generate twiddle factor
    let twiddle = twiddle_factor(pair_id, stage_size, inverse);
    
    // Apply twiddle factor to second element
    let b_twiddle = complex_mul(b, twiddle);
    
    // Butterfly operation
    let result_a = complex_add(a, b_twiddle);
    let result_b = complex_sub(a, b_twiddle);
    
    // Store results
    data[base_idx] = result_a;
    data[pair_idx] = result_b;
} 