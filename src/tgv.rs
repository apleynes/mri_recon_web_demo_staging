use web_time::Instant;

use leptos::logging::log;
use ndarray::{azip, par_azip, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip};
use num_complex::{Complex, ComplexFloat};
use rayon::{array, prelude::*};
use crate::fft::*;
use wgpu::*;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use num_cpus;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TgvParams {
    width: u32,
    height: u32,
    lambda: f32,
    alpha0: f32,
    alpha1: f32,
    tau: f32,
    sigma: f32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExtrapolationParams {
    width: u32,
    height: u32,
    theta: f32,
    _padding: u32,
}

pub struct GpuTgvContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    gradient_pipeline: Option<ComputePipeline>,
    divergence_pipeline: Option<ComputePipeline>,
    sym_gradient_pipeline: Option<ComputePipeline>,
    sym_divergence_pipeline: Option<ComputePipeline>,
    projection_p_pipeline: Option<ComputePipeline>,
    projection_q_pipeline: Option<ComputePipeline>,
    update_u_pipeline: Option<ComputePipeline>,
    update_w_pipeline: Option<ComputePipeline>,
    update_p_pipeline: Option<ComputePipeline>,
    update_q_pipeline: Option<ComputePipeline>,
    extrapolation_u_pipeline: Option<ComputePipeline>,
    extrapolation_w_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
}

impl GpuTgvContext {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No WebGPU adapter found")?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_webgl2_defaults(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            gradient_pipeline: None,
            divergence_pipeline: None,
            sym_gradient_pipeline: None,
            sym_divergence_pipeline: None,
            projection_p_pipeline: None,
            projection_q_pipeline: None,
            update_u_pipeline: None,
            update_w_pipeline: None,
            update_p_pipeline: None,
            update_q_pipeline: None,
            extrapolation_u_pipeline: None,
            extrapolation_w_pipeline: None,
            bind_group_layout: None,
        })
    }

    fn init_pipelines(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Gradient compute shader - matches CPU implementation exactly
        let gradient_shader_source = r#"
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Gradient x with proper periodic boundary handling
                let next_x = select(0u, idx.x + 1u, idx.x < (width - 1u));
                let next_x_idx = idx.y * width + next_x;
                let grad_x = input_data[next_x_idx] - input_data[linear_idx];
                
                // Gradient y with proper periodic boundary handling
                let next_y = select(0u, idx.y + 1u, idx.y < (height - 1u));
                let next_y_idx = next_y * width + idx.x;
                let grad_y = input_data[next_y_idx] - input_data[linear_idx];
                
                // Store gradients (output has 2 components per pixel)
                output_data[linear_idx * 2u] = grad_x;
                output_data[linear_idx * 2u + 1u] = grad_y;
            }
        "#;

        // Divergence compute shader - matches CPU implementation exactly
        let divergence_shader_source = r#"
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Divergence x with proper periodic boundary handling
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let div_x = input_data[linear_idx * 2u] - input_data[prev_x_idx * 2u];
                
                // Divergence y with proper periodic boundary handling
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_y_idx = prev_y * width + idx.x;
                let div_y = input_data[linear_idx * 2u + 1u] - input_data[prev_y_idx * 2u + 1u];
                
                output_data[linear_idx] = -(div_x + div_y);
            }
        "#;

        // Symmetric gradient compute shader - matches CPU implementation exactly
        let sym_gradient_shader_source = r#"
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Input has 2 components per pixel (w vector field)
                let w0_curr = input_data[linear_idx * 2u];
                let w1_curr = input_data[linear_idx * 2u + 1u];
                
                // First diagonal: ∂x w_0 with proper periodic boundary handling
                let next_x = select(0u, idx.x + 1u, idx.x < (width - 1u));
                let next_x_idx = idx.y * width + next_x;
                let w0_next_x = input_data[next_x_idx * 2u];
                let grad_xx = w0_next_x - w0_curr;
                
                // Second diagonal: ∂y w_1 with proper periodic boundary handling
                let next_y = select(0u, idx.y + 1u, idx.y < (height - 1u));
                let next_y_idx = next_y * width + idx.x;
                let w1_next_y = input_data[next_y_idx * 2u + 1u];
                let grad_yy = w1_next_y - w1_curr;
                
                // Off-diagonals: 0.5*(∂y w_0 + ∂x w_1) with proper boundary handling
                let w0_next_y = input_data[next_y_idx * 2u];
                let w1_next_x = input_data[next_x_idx * 2u + 1u];
                let grad_xy = 0.5 * ((w0_next_y - w0_curr) + (w1_next_x - w1_curr));
                
                // Store symmetric gradient (output has 3 components per pixel)
                output_data[linear_idx * 3u] = grad_xx;
                output_data[linear_idx * 3u + 1u] = grad_yy;
                output_data[linear_idx * 3u + 2u] = grad_xy;
            }
        "#;

        // Symmetric divergence compute shader - matches CPU implementation exactly
        let sym_divergence_shader_source = r#"
            @group(0) @binding(0) var<storage, read> input_data: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
            @group(0) @binding(2) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Input has 3 components per pixel (q tensor field)
                let q0_curr = input_data[linear_idx * 3u];     // q_xx
                let q1_curr = input_data[linear_idx * 3u + 1u]; // q_yy  
                let q2_curr = input_data[linear_idx * 3u + 2u]; // q_xy
                
                // First component: -∂x q_0 - 0.5*∂y q_2 with proper boundary handling
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let q0_prev_x = input_data[prev_x_idx * 3u];
                let q2_prev_x = input_data[prev_x_idx * 3u + 2u];
                
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_y_idx = prev_y * width + idx.x;
                let q2_prev_y = input_data[prev_y_idx * 3u + 2u];
                
                let div_x = -(q0_curr - q0_prev_x) - 0.5 * (q2_curr - q2_prev_y);
                
                // Second component: -∂y q_1 - 0.5*∂x q_2 with proper boundary handling
                let q1_prev_y = input_data[prev_y_idx * 3u + 1u];
                let div_y = -(q1_curr - q1_prev_y) - 0.5 * (q2_curr - q2_prev_x);
                
                // Store divergence (output has 2 components per pixel)
                output_data[linear_idx * 2u] = div_x;
                output_data[linear_idx * 2u + 1u] = div_y;
            }
        "#;

        // Update p shader (p = p + sigma * (grad_u_bar - w_bar))
        let update_p_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> p_data: array<f32>;
            @group(0) @binding(1) var<storage, read> grad_data: array<f32>;
            @group(0) @binding(2) var<storage, read> w_bar_data: array<f32>;
            @group(0) @binding(3) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Update p = p + sigma * (grad_u_bar - w_bar)
                p_data[linear_idx * 2u] += params.sigma * (grad_data[linear_idx * 2u] - w_bar_data[linear_idx * 2u]);
                p_data[linear_idx * 2u + 1u] += params.sigma * (grad_data[linear_idx * 2u + 1u] - w_bar_data[linear_idx * 2u + 1u]);
            }
        "#;

        // Update q shader (q = q + sigma * lambda * sym_grad_w_bar)
        let update_q_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> q_data: array<f32>;
            @group(0) @binding(1) var<storage, read> sym_grad_data: array<f32>;
            @group(0) @binding(2) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Update q = q + sigma * lambda * sym_grad_w_bar
                let factor = params.sigma * params.lambda;
                q_data[linear_idx * 3u] += factor * sym_grad_data[linear_idx * 3u];
                q_data[linear_idx * 3u + 1u] += factor * sym_grad_data[linear_idx * 3u + 1u];
                q_data[linear_idx * 3u + 2u] += factor * sym_grad_data[linear_idx * 3u + 2u];
            }
        "#;

        // Projection shader for p (2-component vectors)
        let projection_p_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @group(0) @binding(1) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let p0 = data[linear_idx * 2u];
                let p1 = data[linear_idx * 2u + 1u];
                
                let norm = sqrt(p0 * p0 + p1 * p1);
                let factor = max(1.0, norm / params.alpha0);
                
                data[linear_idx * 2u] = p0 / factor;
                data[linear_idx * 2u + 1u] = p1 / factor;
            }
        "#;

        // Projection shader for q (3-component tensors)
        let projection_q_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @group(0) @binding(1) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let q0 = data[linear_idx * 3u];
                let q1 = data[linear_idx * 3u + 1u];
                let q2 = data[linear_idx * 3u + 2u];
                
                let norm = sqrt(q0 * q0 + q1 * q1 + q2 * q2);
                let factor = max(1.0, norm / params.alpha1);
                
                data[linear_idx * 3u] = q0 / factor;
                data[linear_idx * 3u + 1u] = q1 / factor;
                data[linear_idx * 3u + 2u] = q2 / factor;
            }
        "#;

        // Update shader for u (data fidelity + TGV term)
        let update_u_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> u_data: array<f32>;
            @group(0) @binding(1) var<storage, read> p_data: array<f32>;
            @group(0) @binding(2) var<storage, read> fft_residual: array<f32>;
            @group(0) @binding(3) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Divergence of p with proper periodic boundary handling
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let prev_y_idx = prev_y * width + idx.x;
                
                let div_x = p_data[linear_idx * 2u] - p_data[prev_x_idx * 2u];
                let div_y = p_data[linear_idx * 2u + 1u] - p_data[prev_y_idx * 2u + 1u];
                let div_p = -(div_x + div_y);
                
                // Update u
                u_data[linear_idx] -= params.tau * (params.lambda * div_p + fft_residual[linear_idx]);
            }
        "#;

        // Update shader for w
        let update_w_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> w_data: array<f32>;
            @group(0) @binding(1) var<storage, read> p_data: array<f32>;
            @group(0) @binding(2) var<storage, read> sym_div_q: array<f32>;
            @group(0) @binding(3) var<uniform> params: TgvParams;

            struct TgvParams {
                width: u32,
                height: u32,
                lambda: f32,
                alpha0: f32,
                alpha1: f32,
                tau: f32,
                sigma: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Update w
                w_data[linear_idx * 2u] -= params.tau * (-p_data[linear_idx * 2u] + sym_div_q[linear_idx * 2u]);
                w_data[linear_idx * 2u + 1u] -= params.tau * (-p_data[linear_idx * 2u + 1u] + sym_div_q[linear_idx * 2u + 1u]);
            }
        "#;

        // Extrapolation shader for u_bar and w_bar  
        let extrapolation_u_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> u_bar_data: array<f32>;
            @group(0) @binding(1) var<storage, read> u_data: array<f32>;
            @group(0) @binding(2) var<storage, read> u_old_data: array<f32>;
            @group(0) @binding(3) var<uniform> params: ExtrapolationParams;

            struct ExtrapolationParams {
                width: u32,
                height: u32,
                theta: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // u_bar = u + theta * (u - u_old)
                u_bar_data[linear_idx] = u_data[linear_idx] + params.theta * (u_data[linear_idx] - u_old_data[linear_idx]);
            }
        "#;

        let extrapolation_w_shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> w_bar_data: array<f32>;
            @group(0) @binding(1) var<storage, read> w_data: array<f32>;
            @group(0) @binding(2) var<storage, read> w_old_data: array<f32>;
            @group(0) @binding(3) var<uniform> params: ExtrapolationParams;

            struct ExtrapolationParams {
                width: u32,
                height: u32,
                theta: f32,
                _padding: u32,
            }

            @compute @workgroup_size(16, 16)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.xy;
                let width = params.width;
                let height = params.height;
                
                // Strict bounds checking to prevent out-of-bounds access
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // w_bar = w + theta * (w - w_old)  
                w_bar_data[linear_idx * 2u] = w_data[linear_idx * 2u] + params.theta * (w_data[linear_idx * 2u] - w_old_data[linear_idx * 2u]);
                w_bar_data[linear_idx * 2u + 1u] = w_data[linear_idx * 2u + 1u] + params.theta * (w_data[linear_idx * 2u + 1u] - w_old_data[linear_idx * 2u + 1u]);
            }
        "#;

        // Create shader modules
        let gradient_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gradient Compute Shader"),
            source: ShaderSource::Wgsl(gradient_shader_source.into()),
        });

        let divergence_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Divergence Compute Shader"),
            source: ShaderSource::Wgsl(divergence_shader_source.into()),
        });

        let sym_gradient_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Symmetric Gradient Compute Shader"),
            source: ShaderSource::Wgsl(sym_gradient_shader_source.into()),
        });

        let sym_divergence_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Symmetric Divergence Compute Shader"),
            source: ShaderSource::Wgsl(sym_divergence_shader_source.into()),
        });

        let update_p_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Update P Compute Shader"),
            source: ShaderSource::Wgsl(update_p_shader_source.into()),
        });

        let update_q_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Update Q Compute Shader"),
            source: ShaderSource::Wgsl(update_q_shader_source.into()),
        });

        let projection_p_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Projection P Compute Shader"),
            source: ShaderSource::Wgsl(projection_p_shader_source.into()),
        });

        let projection_q_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Projection Q Compute Shader"),
            source: ShaderSource::Wgsl(projection_q_shader_source.into()),
        });

        let update_u_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Update U Compute Shader"),
            source: ShaderSource::Wgsl(update_u_shader_source.into()),
        });

        let update_w_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Update W Compute Shader"),
            source: ShaderSource::Wgsl(update_w_shader_source.into()),
        });

        let extrapolation_u_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Extrapolation U Compute Shader"),
            source: ShaderSource::Wgsl(extrapolation_u_shader_source.into()),
        });

        let extrapolation_w_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Extrapolation W Compute Shader"),
            source: ShaderSource::Wgsl(extrapolation_w_shader_source.into()),
        });

        // Create bind group layout for 3-input operations (input, output, params)
        let bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("TGV Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group layout for projection operations (data + params)
        let projection_bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Projection Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group layout for 4-input update operations 
        let update_bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Update Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group layout for update_q (3 inputs: q_data, sym_grad_data, params)
        let update_q_bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Update Q Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layouts
        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("TGV Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let projection_pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Projection Pipeline Layout"),
            bind_group_layouts: &[&projection_bind_group_layout],
            push_constant_ranges: &[],
        });

        let update_pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Update Pipeline Layout"),
            bind_group_layouts: &[&update_bind_group_layout],
            push_constant_ranges: &[],
        });

        let update_q_pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Update Q Pipeline Layout"),
            bind_group_layouts: &[&update_q_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create all pipelines
        self.gradient_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.divergence_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Divergence Pipeline"),
            layout: Some(&pipeline_layout),
            module: &divergence_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.sym_gradient_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Symmetric Gradient Pipeline"),
            layout: Some(&pipeline_layout),
            module: &sym_gradient_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.sym_divergence_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Symmetric Divergence Pipeline"),
            layout: Some(&pipeline_layout),
            module: &sym_divergence_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.update_p_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Update P Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_p_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.update_q_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Update Q Pipeline"),
            layout: Some(&update_q_pipeline_layout),
            module: &update_q_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.projection_p_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Projection P Pipeline"),
            layout: Some(&projection_pipeline_layout),
            module: &projection_p_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.projection_q_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Projection Q Pipeline"),
            layout: Some(&projection_pipeline_layout),
            module: &projection_q_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.update_u_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Update U Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_u_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.update_w_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Update W Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_w_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.extrapolation_u_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Extrapolation U Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &extrapolation_u_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.extrapolation_w_pipeline = Some(self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Extrapolation W Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &extrapolation_w_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        self.bind_group_layout = Some(bind_group_layout);

        Ok(())
    }

    pub async fn tgv_reconstruction_gpu(
        &mut self,
        centered_kspace: &ArrayView2<'_, Complex<f64>>,
        centered_mask: &ArrayView2<'_, f32>,
        lambda: f32,
        alpha1: f32,
        alpha0: f32,
        tau: f32,
        sigma: f32,
        max_iter: usize,
    ) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
        if self.gradient_pipeline.is_none() {
            self.init_pipelines()?;
        }

        let (ny, nx) = centered_kspace.dim();
        log!("Starting GPU TGV reconstruction with dimensions: {}x{}", ny, nx);
        
        // Initialize with zero-filled reconstruction on CPU
        let mut masked_kspace = Array2::<Complex<f64>>::zeros((ny, nx));
        par_azip!((x in &mut masked_kspace, &y in centered_kspace, &z in centered_mask) {
            if z > 0. {
                *x = y;
            }
        });
        
        let masked_kspace_view = masked_kspace.view();
        let shifted_masked_kspace = ifft2shift(&masked_kspace_view);
        let u_complex = ifft2(&shifted_masked_kspace.view());
        let u_data: Vec<f32> = u_complex.into_iter().map(|x| x.re as f32).collect();
        
        // Create GPU buffers
        let u_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("U Buffer"),
            contents: bytemuck::cast_slice(&u_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let w_data = vec![0.0f32; ny * nx * 2];
        let w_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("W Buffer"),
            contents: bytemuck::cast_slice(&w_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let p_data = vec![0.0f32; ny * nx * 2];
        let p_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("P Buffer"),
            contents: bytemuck::cast_slice(&p_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let q_data = vec![0.0f32; ny * nx * 3];
        let q_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Q Buffer"),
            contents: bytemuck::cast_slice(&q_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Create u_bar and w_bar buffers (copies of u and w)
        let u_bar_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("U Bar Buffer"),
            contents: bytemuck::cast_slice(&u_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let w_bar_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("W Bar Buffer"),
            contents: bytemuck::cast_slice(&w_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Create u_old and w_old buffers for extrapolation 
        let u_old_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("U Old Buffer"),
            contents: bytemuck::cast_slice(&u_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let w_old_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("W Old Buffer"),
            contents: bytemuck::cast_slice(&w_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // Create parameter buffer
        let params = TgvParams {
            width: nx as u32,
            height: ny as u32,
            lambda,
            alpha0,
            alpha1,
            tau,
            sigma,
            _padding: 0,
        };
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // Create temporary buffers for intermediate results
        let grad_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Gradient Buffer"),
            size: (ny * nx * 2 * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sym_grad_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Symmetric Gradient Buffer"),
            size: (ny * nx * 3 * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sym_div_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Symmetric Divergence Buffer"),
            size: (ny * nx * 2 * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let div_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Divergence Buffer"),
            size: (ny * nx * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let fft_residual_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("FFT Residual Buffer"),
            size: (ny * nx * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create readback buffers
        let u_readback_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("U Readback Buffer"),
            size: (ny * nx * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let _u_staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("U Staging Buffer"),
            size: (ny * nx * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Get bind group layouts
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();
        let projection_bind_group_layout = &self.projection_p_pipeline.as_ref().unwrap().get_bind_group_layout(0);
        let update_bind_group_layout = &self.update_u_pipeline.as_ref().unwrap().get_bind_group_layout(0);
        let update_q_bind_group_layout = &self.update_q_pipeline.as_ref().unwrap().get_bind_group_layout(0);

        // Use floating point ceil to calculate workgroup counts
        let workgroup_count_x = f32::ceil(nx as f32 / 16.0) as u32;
        let workgroup_count_y = f32::ceil(ny as f32 / 16.0) as u32;

        // Initialize momentum variable
        let mut t: f32 = 1.0;

        // Main TGV iteration loop
        for iteration in 0..max_iter {
            log!("GPU TGV iteration {}/{}", iteration + 1, max_iter);
            
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("TGV Iteration"),
            });

            // 1. Compute gradient of u_bar
            {
                let gradient_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Gradient Bind Group"),
                    layout: bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: u_bar_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: grad_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Gradient Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.gradient_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &gradient_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 2. Update p = p + sigma * (grad_u_bar - w_bar) and project
            {
                let update_p_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Update P Bind Group"),
                    layout: update_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: p_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: grad_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: w_bar_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Update P Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.update_p_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &update_p_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // Project p
            {
                let projection_p_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Projection P Bind Group"),
                    layout: projection_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: p_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Projection P Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.projection_p_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &projection_p_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 3. Compute symmetric gradient of w_bar
            {
                let sym_gradient_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Symmetric Gradient Bind Group"),
                    layout: bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: w_bar_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: sym_grad_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Symmetric Gradient Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.sym_gradient_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &sym_gradient_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 4. Update q = q + sigma * lambda * sym_grad_w_bar
            {
                let update_q_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Update Q Bind Group"),
                    layout: update_q_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: q_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: sym_grad_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Update Q Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.update_q_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &update_q_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 5. Project q
            {
                let projection_q_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Projection Q Bind Group"),
                    layout: projection_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: q_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Projection Q Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.projection_q_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &projection_q_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // Submit dual updates
            self.queue.submit(Some(encoder.finish()));

            // *** CRITICAL FIX: Save old values AFTER dual updates but BEFORE primal updates ***
            // This matches the CPU implementation exactly and is essential for correct convergence
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Save Old Values"),
            });
            encoder.copy_buffer_to_buffer(&u_buffer, 0, &u_old_buffer, 0, u_buffer.size());
            encoder.copy_buffer_to_buffer(&w_buffer, 0, &w_old_buffer, 0, w_buffer.size());
            self.queue.submit(Some(encoder.finish()));

            // 6. Compute divergence of p for primal updates
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Primal Updates"),
            });
            
            {
                let divergence_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Divergence Bind Group"),
                    layout: bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: p_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: div_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Divergence Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.divergence_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &divergence_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // Submit divergence computation
            self.queue.submit(Some(encoder.finish()));

            // 7. Handle data fidelity term on CPU (FFT operations)
            // Copy u from GPU to CPU
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Copy U to CPU"),
            });
            encoder.copy_buffer_to_buffer(&u_buffer, 0, &u_readback_buffer, 0, u_readback_buffer.size());
            self.queue.submit(Some(encoder.finish()));

            // Wait for copy to complete and read data
            let buffer_slice = u_readback_buffer.slice(..);
            let (sender, receiver) = futures_channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            receiver.await.unwrap()?;

            let data_slice = buffer_slice.get_mapped_range();
            let u_cpu_data: &[f32] = bytemuck::cast_slice(&data_slice);
            
            // Perform FFT operations on CPU
            let complex_u = Array2::<Complex<f64>>::from_shape_vec(
                (ny, nx), 
                u_cpu_data.iter().map(|&x| Complex::new(x as f64, 0.0)).collect()
            ).unwrap();
            let fft_u = fft2shift(&fft2(&complex_u.view()).view());
            let mut residual = Array2::<Complex<f64>>::zeros((ny, nx));
            par_azip!((x in &mut residual, &y in &fft_u, &z in centered_kspace, &w in centered_mask) {
                if w > 0. {
                    *x = y - z;
                }
            });
            let ifft_residual = ifft2(&ifft2shift(&residual.view()).view());
            let ifft_residual_data: Vec<f32> = ifft_residual.into_iter().map(|x| x.re as f32).collect();

            // Unmap the buffer
            drop(data_slice);
            u_readback_buffer.unmap();

            // Copy FFT residual back to GPU
            self.queue.write_buffer(&fft_residual_buffer, 0, bytemuck::cast_slice(&ifft_residual_data));

            // 8. Update u using divergence and data fidelity term
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Update U"),
            });

            {
                let update_u_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Update U Bind Group"),
                    layout: update_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: u_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: div_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: fft_residual_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Update U Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.update_u_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &update_u_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 9. Compute symmetric divergence of q
            {
                let sym_divergence_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Symmetric Divergence Bind Group"),
                    layout: bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: q_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: sym_div_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Symmetric Divergence Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.sym_divergence_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &sym_divergence_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // 10. Update w
            {
                let update_w_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Update W Bind Group"),
                    layout: update_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: w_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: p_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: sym_div_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Update W Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.update_w_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &update_w_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // Submit all primal updates
            self.queue.submit(Some(encoder.finish()));

            // 11. Update u_bar and w_bar (proper extrapolation with correct old values)
            let t_old = t;
            t = (1. + (1. + 4. * t_old.powi(2)).sqrt()) / 2.0;
            let theta = (1. - t_old) / t;
            
            // Create extrapolation parameter buffer
            let extrapolation_params = ExtrapolationParams {
                width: nx as u32,
                height: ny as u32,
                theta,
                _padding: 0,
            };
            let extrapolation_params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Extrapolation Params Buffer"),
                contents: bytemuck::cast_slice(&[extrapolation_params]),
                usage: BufferUsages::UNIFORM,
            });

            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Extrapolation"),
            });

            // Extrapolate u_bar
            {
                let extrapolation_u_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Extrapolation U Bind Group"),
                    layout: update_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: u_bar_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: u_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: u_old_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: extrapolation_params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Extrapolation U Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.extrapolation_u_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &extrapolation_u_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            // Extrapolate w_bar
            {
                let extrapolation_w_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Extrapolation W Bind Group"),
                    layout: update_bind_group_layout,
                    entries: &[
                        BindGroupEntry { binding: 0, resource: w_bar_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 1, resource: w_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 2, resource: w_old_buffer.as_entire_binding() },
                        BindGroupEntry { binding: 3, resource: extrapolation_params_buffer.as_entire_binding() },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Extrapolation W Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(self.extrapolation_w_pipeline.as_ref().unwrap());
                compute_pass.set_bind_group(0, &extrapolation_w_bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
            }

            self.queue.submit(Some(encoder.finish()));
        }

        // Final readback
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Final Readback"),
        });
        encoder.copy_buffer_to_buffer(&u_buffer, 0, &u_readback_buffer, 0, u_readback_buffer.size());
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = u_readback_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap()?;

        let data_slice = buffer_slice.get_mapped_range();
        let final_data: &[f32] = bytemuck::cast_slice(&data_slice);
        let result = Array2::<f32>::from_shape_vec((ny, nx), final_data.to_vec()).unwrap();

        drop(data_slice);
        u_readback_buffer.unmap();

        log!("GPU TGV reconstruction completed");
        Ok(result)
    }
}

// High-level API with automatic GPU/CPU fallback
pub async fn tgv_mri_reconstruction_auto(
    centered_kspace: &ArrayView2<'_, Complex<f64>>,
    centered_mask: &ArrayView2<'_, f32>,
    lambda: f32,
    alpha1: f32,
    alpha0: f32,
    tau: f32,
    sigma: f32,
    max_iter: usize,
) -> Array2<f32> {
    match GpuTgvContext::new().await {
        Ok(mut ctx) => {
            match ctx.tgv_reconstruction_gpu(
                centered_kspace,
                centered_mask,
                lambda,
                alpha1,
                alpha0,
                tau,
                sigma,
                max_iter,
            ).await {
                Ok(result) => {
                    log!("Using GPU TGV reconstruction");
                    result
                },
                Err(_) => {
                    log!("GPU TGV failed, falling back to CPU");
                    tgv_mri_reconstruction(
                        centered_kspace,
                        centered_mask,
                        lambda,
                        alpha1,
                        alpha0,
                        tau,
                        sigma,
                        max_iter,
                    )
                }
            }
        },
        Err(_) => {
            log!("WebGPU not available, using CPU TGV reconstruction");
            tgv_mri_reconstruction(
                centered_kspace,
                centered_mask,
                lambda,
                alpha1,
                alpha0,
                tau,
                sigma,
                max_iter,
            )
        }
    }
}

fn roll1d(a: &ArrayView1<f32>, roll_amount: i32) -> Array1<f32> {
    
    ndarray::concatenate![Axis(0), a.slice(s![-roll_amount..]), a.slice(s![..-roll_amount])]
}

fn roll2d(a: &ArrayView2<f32>, axis: usize, roll_amount: i32) -> Array2<f32> {
    assert!(roll_amount.abs() > 0);
    if axis == 0 {
        return ndarray::concatenate![Axis(0), a.slice(s![-roll_amount.., ..]), a.slice(s![..-roll_amount, ..])]
    } else if axis == 1 {
        return ndarray::concatenate![Axis(1), a.slice(s![.., -roll_amount..]), a.slice(s![.., ..-roll_amount,])]
    } else {
        return a.to_owned()
    }
}

fn gradient(u: &ArrayView2<f32>) -> Array3<f32> {
    // let grad_x = roll2d(&u.view(), 1, -1) - u;
    let num_cpus = num_cpus::get();
    let (y_size, x_size) = u.dim();

    let chunk_size = y_size / num_cpus;
    let mut grad_x = u.clone().to_owned();
    grad_x.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });
    
    // let grad_y = roll2d(&u.view(), 0, -1) - u;

    let chunk_size = x_size / num_cpus;
    let mut grad_y = u.clone().to_owned();
    grad_y.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    ndarray::stack![Axis(2), grad_x, grad_y]
}

fn divergence(p: &ArrayView3<f32>) -> Array2<f32> {
    // let first_term = p.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&p.slice(s![.., .., 0]), 1, 1);

    // Calculate first term in parallel
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = p.dim();
    let chunk_size = y_size / num_cpus;
    let mut first_term = p.slice(s![.., .., 0]).clone().to_owned();
    first_term.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff);
        });
    });

    // let second_term = p.slice(s![.., .., 1]).to_owned() 
    // - roll2d(&p.slice(s![.., .., 1]), 0, 1);

    // Calculate second term in parallel
    let chunk_size = x_size / num_cpus;
    let mut second_term = p.slice(s![.., .., 1]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff);
        });
    });
    // -(first_term + second_term)
    // Calculate output
    let chunk_size = y_size / num_cpus;
    let mut first_term_chunks = first_term.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    first_term_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        // x = -(x + y);
        x.assign(&(-(x.to_owned() + y)));
    });
    first_term

}

fn sym_gradient(w: &ArrayView3<f32>) -> Array3<f32> {
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = w.dim();

    // Calculate first diagonal: ∂x w_0 (shift right, forward difference)
    let chunk_size = y_size / num_cpus;
    let mut first_diagonal = w.slice(s![.., .., 0]).clone().to_owned();
    first_diagonal.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0 as usize] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });

    // Calculate second diagonal: ∂y w_1 (shift down, forward difference)
    let chunk_size = x_size / num_cpus;
    let mut second_diagonal = w.slice(s![.., .., 1]).clone().to_owned();
    second_diagonal.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0 as usize] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    // Calculate off-diagonals: 0.5*(∂y w_0 + ∂x w_1)
    // tmp1: ∂y w_0 (shift down, forward difference)
    let chunk_size = x_size / num_cpus;
    let mut tmp1 = w.slice(s![.., .., 0]).clone().to_owned();
    tmp1.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0 as usize] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    // tmp2: ∂x w_1 (shift right, forward difference)
    let chunk_size = y_size / num_cpus;
    let mut tmp2 = w.slice(s![.., .., 1]).clone().to_owned();
    tmp2.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0 as usize] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });

    let mut off_diagonals = tmp1;
    let mut off_diagonals_chunks = off_diagonals.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let tmp2_chunks = tmp2.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    off_diagonals_chunks.zip(tmp2_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(0.5 * (x.to_owned() + y)));
    });

    ndarray::stack![Axis(2), first_diagonal, second_diagonal, off_diagonals]
}

fn sym_divergence(q: &ArrayView3<f32>) -> Array3<f32> {
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = q.dim();

    // First component: ∂x q_0 - ∂y q_2
    // ∂x q_0: backward difference (shift left)
    let chunk_size = y_size / num_cpus;
    let mut first_component = q.slice(s![.., .., 0]).clone().to_owned();
    first_component.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff.map(|x| -x));
        });
    });

    // ∂y q_2: backward difference (shift up)
    let chunk_size = x_size / num_cpus;
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff.map(|x| -0.5 * x));
        });
    });
    let mut first_component_chunks = first_component.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    first_component_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() + y));
    });

    // Second component: ∂y q_1 - ∂x q_2
    // ∂y q_1: backward difference (shift up)
    let chunk_size = x_size / num_cpus;
    let mut second_component = q.slice(s![.., .., 1]).clone().to_owned();
    second_component.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff.map(|x| -x));
        });
    });

    // ∂x q_2: backward difference (shift left)
    let chunk_size = y_size / num_cpus;
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff.map(|x| -0.5 * x));
        });
    });
    let mut second_component_chunks = second_component.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    second_component_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() + y));
    });

    ndarray::stack![Axis(2), first_component, second_component]
}

fn proj_p(p: &ArrayView3<f32>, alpha1: &f32) -> Array3<f32> {
    let norm = p.clone().to_owned().map_axis_mut(Axis(2), |slice| {
        slice.powi(2).sum().sqrt()
    });

    let mut factor = norm;
    factor.par_iter_mut().for_each(|x| {
        if (*x / alpha1) > 1. {
            *x = *x / alpha1;
        } else {
            *x = 1.0;
        }
    });
    let mut p_proj = p.to_owned();
    
    // Apply chunked parallelization for slice1
    let num_cpus = num_cpus::get();
    let (y_size, _, _) = p_proj.dim();
    let chunk_size = y_size / num_cpus;
    let mut sliced_p_proj = p_proj.slice_mut(s![.., .., 0]);
    let mut slice1_chunks = sliced_p_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice1_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Apply chunked parallelization for slice2
    let mut sliced_p_proj = p_proj.slice_mut(s![.., .., 1]);
    let mut slice2_chunks = sliced_p_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice2_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    p_proj
}

fn proj_q(q: &ArrayView3<f32>, alpha0: &f32) -> Array3<f32> {
    let norm = q.clone().to_owned().map_axis_mut(Axis(2), |slice| {
        slice.powi(2).sum().sqrt()
    });

    let mut factor = norm;
    factor.par_iter_mut().for_each(|x| {
        if (*x / alpha0) > 1. {
            *x = *x / alpha0;
        } else {
            *x = 1.0;
        }
    });
    let mut q_proj = q.to_owned();
    
    // Apply chunked parallelization for all three slices
    let num_cpus = num_cpus::get();
    let (y_size, _, _) = q_proj.dim();
    let chunk_size = y_size / num_cpus;
    
    // Slice 0
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 0]);
    let mut slice1_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice1_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Slice 1
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 1]);
    let mut slice2_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice2_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Slice 2
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 2]);
    let mut slice3_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice3_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    q_proj
}


// pub struct TGV {
//     pub lambda: f32,
//     pub mu: f32,
//     pub nu: f32,
// }

// impl TGV {
//     pub fn new(lambda: f32, mu: f32, nu: f32) -> Self {
//         Self { lambda, mu, nu }
//     }
    
// }

// enum ArrayInputType {
//     Array2(Array2<f32>),
//     Array3(Array3<f32>),
// }

// fn report_min_max(array: ArrayInputType) {
//     log!("Min max: {}, {}", array.clone().into_iter().reduce(f32::min).unwrap(), array.clone().into_iter().reduce(f32::max).unwrap());
// }

pub fn tgv_mri_reconstruction(
    centered_kspace: &ArrayView2<Complex<f64>>,
    centered_mask: &ArrayView2<f32>,
    lambda: f32,
    alpha1: f32,
    alpha0: f32,
    tau: f32,
    sigma: f32,
    max_iter: usize,
) -> Array2<f32> {

    // Start timing
    let start_time = Instant::now();

    let (ny, nx) = centered_kspace.dim();
    // Initialize with zero-filled reconstruction
    let mut masked_kspace = Array2::<Complex<f64>>::zeros((ny, nx));
    par_azip!((x in &mut masked_kspace, &y in centered_kspace, &z in centered_mask) {
        if z > 0. {
            *x = y;
        }
    });
    let masked_kspace_view = masked_kspace.view();
    let shifted_masked_kspace = ifft2shift(&masked_kspace_view);
    let u = ifft2(&shifted_masked_kspace.view());
    let mut u = Array2::<f32>::from_shape_vec((ny, nx), u.into_iter().map(|x| x.re as f32).collect()).unwrap();
    
    // Initialize TGV variables
    let mut w = Array3::<f32>::zeros((ny, nx, 2));
    let mut p = Array3::<f32>::zeros((ny, nx, 2));
    let mut q = Array3::<f32>::zeros((ny, nx, 3));

    let mut u_bar = u.clone();
    let mut w_bar = w.clone();

    let t: f32 = 1.0;  // Adaptive momentum

    let num_cpus = num_cpus::get();
    let (y_size, _, _) = p.dim();
    let chunk_size = y_size / num_cpus;
    for i in 0..max_iter {
        let grad_u_bar = gradient(&u_bar.view());
        
        // Replace par_azip with chunked parallelization

        let mut p_chunks = p.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let grad_chunks = grad_u_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let w_bar_chunks = w_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        p_chunks.zip(grad_chunks.zip(w_bar_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(x.to_owned() + sigma * (y.to_owned() - z.to_owned())));
        });
        
        let p = proj_p(&p.view(), &(&alpha0));

        let sym_grad_w_bar = sym_gradient(&w_bar.view());
        
        // Replace par_azip with chunked parallelization
        let mut q_chunks = q.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let sym_grad_chunks = sym_grad_w_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        q_chunks.zip(sym_grad_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() + sigma * lambda * y.to_owned()));
        });
        
        let q = proj_q(&q.view(), &(&alpha1));

        // Primal updates
        let u_old = u.clone();
        let w_old = w.clone();

        // Update u: TGV + data fidelity
        let div_p = divergence(&p.view());
        
        // Replace par_azip with chunked parallelization for u update
        let mut u_chunks = u.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let div_p_chunks = div_p.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_chunks.zip(div_p_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() - tau * lambda * y.to_owned()));
        });

        // Data fidelity term
        let complex_u = Array2::<Complex<f64>>::from_shape_vec((ny, nx), (&u).into_iter().map(|x| Complex::new(*x as f64, 0.0)).collect()).unwrap();
        let fft_u = fft2shift(&fft2(&complex_u.view()).view());
        let mut residual = Array2::<Complex<f64>>::zeros((ny, nx));
        
        // Replace par_azip with chunked parallelization
        let mut residual_chunks = residual.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let fft_u_chunks = fft_u.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let centered_kspace_chunks = centered_kspace.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let centered_mask_chunks = centered_mask.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        residual_chunks.zip(fft_u_chunks.zip(centered_kspace_chunks.zip(centered_mask_chunks))).par_bridge().for_each(|(mut x, (y, (z, w)))| {
            for ((x_elem, y_elem), (z_elem, w_elem)) in x.iter_mut().zip(y.iter()).zip(z.iter().zip(w.iter())) {
                if *w_elem > 0. {
                    *x_elem = *y_elem - *z_elem;
                }
            }
        });
        
        let ifft_residual = ifft2(&ifft2shift(&residual.view()).view());
        let ifft_residual = Array2::<f32>::from_shape_vec((ny, nx), ifft_residual.into_iter().map(|x| x.re as f32).collect()).unwrap();
        // Replace par_azip with chunked parallelization for final u update
        let mut u_chunks = u.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let ifft_residual_chunks = ifft_residual.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_chunks.zip(ifft_residual_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() - tau * y.to_owned()));
        });

        // Update w
        let sym_div_q = sym_divergence(&q.view());
        
        // Replace par_azip with chunked parallelization for w update
        let mut w_chunks = w.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let p_chunks = p.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let sym_div_q_chunks = sym_div_q.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        w_chunks.zip(p_chunks.zip(sym_div_q_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(x.to_owned() - tau * (-y.to_owned() + z.to_owned())));
        });
        
        // Extrapolation
        // Update momentum
        let t_old = t;
        let t = (1. + (1. + 4. * t_old.powi(2)).sqrt()) / 2.0;
        let theta = (1. - t_old) / t;
        
        // Replace par_azip with chunked parallelization for u_bar update
        let mut u_bar_chunks = u_bar.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let u_chunks = u.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let u_old_chunks = u_old.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_bar_chunks.zip(u_chunks.zip(u_old_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(y.to_owned() + theta * (y.to_owned() - z.to_owned())));
        });
        
        // Replace par_azip with chunked parallelization for w_bar update
        let mut w_bar_chunks = w_bar.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let w_chunks = w.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let w_old_chunks = w_old.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        w_bar_chunks.zip(w_chunks.zip(w_old_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(y.to_owned() + theta * (y.to_owned() - z.to_owned())));
        });

        let total_residual: f64 = residual.map(|x| x.re.powi(2)).sum();
        // log!("Iteration: {}, Total residual: {:.3e}", i, total_residual);


        // log!("Min max: {}, {}", u.clone().into_iter().reduce(f32::min).unwrap(), u.clone().into_iter().reduce(f32::max).unwrap());
    }
    // Convert to real
    // let u = Array2::<f32>::from_shape_vec((ny, nx), u.into_iter().map(|x| x.re as f32).collect()).unwrap();
    // log!("Min max: {}, {}", u.clone().into_iter().reduce(f32::min).unwrap(), u.clone().into_iter().reduce(f32::max).unwrap());

    let end_time = Instant::now();
    log!("Time taken: {:?}", end_time.duration_since(start_time));
    u
}

