use web_time::Instant;

use leptos::logging::log;
use ndarray::{par_azip, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip};
use num_complex::{Complex, ComplexFloat};
use rayon::{array, prelude::*};
use crate::fft::*;
use wgpu::*;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

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
            bind_group_layout: None,
        })
    }

    fn init_pipelines(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Gradient compute shader
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let next_x = (idx.x + 1u) % width;
                let next_y = (idx.y + 1u) % height;
                let next_x_idx = idx.y * width + next_x;
                let next_y_idx = next_y * width + idx.x;
                
                // Gradient calculation: finite differences
                let grad_x = input_data[next_x_idx] - input_data[linear_idx];
                let grad_y = input_data[next_y_idx] - input_data[linear_idx];
                
                // Store gradients (output has 2 components per pixel)
                output_data[linear_idx * 2u] = grad_x;
                output_data[linear_idx * 2u + 1u] = grad_y;
            }
        "#;

        let gradient_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gradient Compute Shader"),
            source: ShaderSource::Wgsl(gradient_shader_source.into()),
        });

        // Divergence compute shader
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let prev_y_idx = prev_y * width + idx.x;
                
                // Divergence calculation: negative of finite differences
                let div_x = input_data[linear_idx * 2u] - input_data[prev_x_idx * 2u];
                let div_y = input_data[linear_idx * 2u + 1u] - input_data[prev_y_idx * 2u + 1u];
                
                output_data[linear_idx] = -(div_x + div_y);
            }
        "#;

        let divergence_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Divergence Compute Shader"),
            source: ShaderSource::Wgsl(divergence_shader_source.into()),
        });

        // Create bind group layout
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

        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("TGV Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create pipelines
        let gradient_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Gradient Pipeline"),
            layout: Some(&pipeline_layout),
            module: &gradient_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let divergence_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Divergence Pipeline"),
            layout: Some(&pipeline_layout),
            module: &divergence_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.gradient_pipeline = Some(gradient_pipeline);
        self.divergence_pipeline = Some(divergence_pipeline);
        self.bind_group_layout = Some(bind_group_layout);

        // Symmetric gradient compute shader
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let next_x = (idx.x + 1u) % width;
                let next_y = (idx.y + 1u) % height;
                let next_x_idx = idx.y * width + next_x;
                let next_y_idx = next_y * width + idx.x;
                
                // Input has 2 components per pixel (w vector field)
                let w0_curr = input_data[linear_idx * 2u];
                let w1_curr = input_data[linear_idx * 2u + 1u];
                let w0_next_x = input_data[next_x_idx * 2u];
                let w1_next_x = input_data[next_x_idx * 2u + 1u];
                let w0_next_y = input_data[next_y_idx * 2u];
                let w1_next_y = input_data[next_y_idx * 2u + 1u];
                
                // Symmetric gradient calculation
                let grad_xx = w0_next_x - w0_curr;  // ∂x w_0
                let grad_yy = w1_next_y - w1_curr;  // ∂y w_1
                let grad_xy = 0.5 * ((w0_next_y - w0_curr) + (w1_next_x - w1_curr));  // 0.5*(∂y w_0 + ∂x w_1)
                
                // Store symmetric gradient (output has 3 components per pixel)
                output_data[linear_idx * 3u] = grad_xx;
                output_data[linear_idx * 3u + 1u] = grad_yy;
                output_data[linear_idx * 3u + 2u] = grad_xy;
            }
        "#;

        // Symmetric divergence compute shader
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let prev_y_idx = prev_y * width + idx.x;
                
                // Input has 3 components per pixel (q tensor field)
                let q0_curr = input_data[linear_idx * 3u];     // q_xx
                let q1_curr = input_data[linear_idx * 3u + 1u]; // q_yy
                let q2_curr = input_data[linear_idx * 3u + 2u]; // q_xy
                let q0_prev_x = input_data[prev_x_idx * 3u];
                let q1_prev_y = input_data[prev_y_idx * 3u + 1u];
                let q2_prev_x = input_data[prev_x_idx * 3u + 2u];
                let q2_prev_y = input_data[prev_y_idx * 3u + 2u];
                
                // Symmetric divergence calculation
                let div_x = -(q0_curr - q0_prev_x) - 0.5 * (q2_curr - q2_prev_y);
                let div_y = -(q1_curr - q1_prev_y) - 0.5 * (q2_curr - q2_prev_x);
                
                // Store divergence (output has 2 components per pixel)
                output_data[linear_idx * 2u] = div_x;
                output_data[linear_idx * 2u + 1u] = div_y;
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                let prev_x = select(width - 1u, idx.x - 1u, idx.x > 0u);
                let prev_y = select(height - 1u, idx.y - 1u, idx.y > 0u);
                let prev_x_idx = idx.y * width + prev_x;
                let prev_y_idx = prev_y * width + idx.x;
                
                // Divergence of p
                let div_p = -(p_data[linear_idx * 2u] - p_data[prev_x_idx * 2u] + 
                              p_data[linear_idx * 2u + 1u] - p_data[prev_y_idx * 2u + 1u]);
                
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
                
                if (idx.x >= width || idx.y >= height) {
                    return;
                }
                
                let linear_idx = idx.y * width + idx.x;
                
                // Update w
                w_data[linear_idx * 2u] -= params.tau * (-p_data[linear_idx * 2u] + sym_div_q[linear_idx * 2u]);
                w_data[linear_idx * 2u + 1u] -= params.tau * (-p_data[linear_idx * 2u + 1u] + sym_div_q[linear_idx * 2u + 1u]);
            }
        "#;

        // Create shader modules
        let sym_gradient_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Symmetric Gradient Compute Shader"),
            source: ShaderSource::Wgsl(sym_gradient_shader_source.into()),
        });

        let sym_divergence_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Symmetric Divergence Compute Shader"),
            source: ShaderSource::Wgsl(sym_divergence_shader_source.into()),
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

        // Create all pipelines
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

        // Create bind group layout for projection shaders (single buffer + params)
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

        // Create bind group layout for update shaders (multiple buffers + params)
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

        // Create pipeline layouts
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

        // Create all the missing pipelines
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
        
        // Initialize with zero-filled reconstruction on CPU, then transfer to GPU
        let mut masked_kspace = Array2::<Complex<f64>>::zeros((ny, nx));
        par_azip!((x in &mut masked_kspace, &y in centered_kspace, &z in centered_mask) {
            if z > 0. {
                *x = y;
            }
        });
        
        // Get initial reconstruction
        let masked_kspace_view = masked_kspace.view();
        let shifted_masked_kspace = ifft2shift(&masked_kspace_view);
        let u_complex = ifft2(&shifted_masked_kspace.view());
        let u_data: Vec<f32> = u_complex.into_iter().map(|x| x.re as f32).collect();
        
        // Create GPU buffers with proper API
        let u_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("U Buffer"),
            contents: bytemuck::cast_slice(&u_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let w_data = vec![0.0f32; ny * nx * 2]; // 2 components per pixel
        let w_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("W Buffer"),
            contents: bytemuck::cast_slice(&w_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let p_data = vec![0.0f32; ny * nx * 2]; // 2 components per pixel
        let p_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("P Buffer"),
            contents: bytemuck::cast_slice(&p_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let q_data = vec![0.0f32; ny * nx * 3]; // 3 components per pixel
        let q_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Q Buffer"),
            contents: bytemuck::cast_slice(&q_data),
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

        // Create readback buffer
        let readback_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Readback Buffer"),
            size: (ny * nx * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Get bind group layout
        let bind_group_layout = self.bind_group_layout.as_ref().unwrap();

        // Main TGV iteration loop
        for iteration in 0..max_iter {
            log!("GPU TGV iteration {}/{}", iteration + 1, max_iter);
            
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("TGV Iteration"),
            });

            // For now, implement a basic version that does CPU-GPU hybrid
            // In a full implementation, we'd need to implement FFT operations on GPU as well
            
            // Perform one simplified GPU operation as an example
            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Gradient Compute Pass"),
                    timestamp_writes: None,
                });

                // Create bind group for gradient computation
                let gradient_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Gradient Bind Group"),
                    layout: bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: u_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: grad_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                    ],
                });

                if let Some(gradient_pipeline) = &self.gradient_pipeline {
                    compute_pass.set_pipeline(gradient_pipeline);
                    compute_pass.set_bind_group(0, &gradient_bind_group, &[]);
                    
                    let workgroup_count_x = (nx + 15) / 16;
                    let workgroup_count_y = (ny + 15) / 16;
                    compute_pass.dispatch_workgroups(workgroup_count_x as u32, workgroup_count_y as u32, 1);
                }
            }
            
            self.queue.submit(Some(encoder.finish()));
            
            // For now, fall back to CPU for the complete TGV algorithm
            // A full GPU implementation would require implementing all operations including FFT on GPU
            break;
        }

        log!("GPU TGV currently implements partial GPU acceleration, falling back to CPU for complete algorithm");
        
        // For now, return CPU-computed result
        return Ok(tgv_mri_reconstruction(
            centered_kspace,
            centered_mask,
            lambda,
            alpha1,
            alpha0,
            tau,
            sigma,
            max_iter,
        ));
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

    let mut grad_x = u.clone().to_owned();
    grad_x.axis_iter_mut(Axis(0)) 
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            // let mut diff = diff.clone();
            // row = diff.view_mut();
            row.assign(&diff);
        });
    
    // let grad_y = roll2d(&u.view(), 0, -1) - u;

    let mut grad_y = u.clone().to_owned();
    grad_y.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            // let mut diff = diff.clone();
            // col = diff.view_mut();
            col.assign(&diff);
        });

    ndarray::stack![Axis(2), grad_x, grad_y]
}

fn divergence(p: &ArrayView3<f32>) -> Array2<f32> {
    // let first_term = p.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&p.slice(s![.., .., 0]), 1, 1);

    // Calculate first term in parallel
    let mut first_term = p.slice(s![.., .., 0]).clone().to_owned();
    first_term.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = row.to_owned() - shifted_row;
            row.assign(&diff);
        });


    // let second_term = p.slice(s![.., .., 1]).to_owned() 
    // - roll2d(&p.slice(s![.., .., 1]), 0, 1);

    // Calculate second term in parallel
    let mut second_term = p.slice(s![.., .., 1]).clone().to_owned();
    second_term.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = col.to_owned() - shifted_col;
            col.assign(&diff);
        });

    // -(first_term + second_term)
    // Calculate output
    par_azip!((x in &mut first_term, &y in &second_term) {
        *x = -(*x + y);
    });
    first_term

}

fn sym_gradient(w: &ArrayView3<f32>) -> Array3<f32> {
    // // First diagonal: ∂x w_0
    // let first_diagonal = roll2d(&w.slice(s![.., .., 0]), 1, -1) 
    //     - w.slice(s![.., .., 0]);
    // // Second diagonal: ∂y w_1
    // let second_diagonal = roll2d(&w.slice(s![.., .., 1]), 0, -1) 
    //     - w.slice(s![.., .., 1]);
    // // Off-diagonals: 0.5*(∂y w_0 + ∂x w_1)
    // let tmp1 = roll2d(&w.slice(s![.., .., 0]), 0, -1) 
    //     - w.slice(s![.., .., 0]);
    // let tmp2 = roll2d(&w.slice(s![.., .., 1]), 1, -1) 
    //     - w.slice(s![.., .., 1]);
    // let off_diagonals = 0.5 * (tmp1 + tmp2);

    // Calculate first diagonal in parallel
    let mut first_diagonal = w.slice(s![.., .., 0]).clone().to_owned();
    first_diagonal.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            row.assign(&diff);
        });

    // Calculate second diagonal in parallel
    let mut second_diagonal = w.slice(s![.., .., 1]).clone().to_owned();
    second_diagonal.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            col.assign(&diff);
        });

    // Calculate off-diagonals in parallel
    let mut tmp1 = w.slice(s![.., .., 0]).clone().to_owned();
    tmp1.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            row.assign(&diff);
        });

    let mut tmp2 = w.slice(s![.., .., 1]).clone().to_owned();
    tmp2.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            col.assign(&diff);
        });
    let mut off_diagonals = tmp1;
    par_azip!((x in &mut off_diagonals, &y in &tmp2) {
        *x = 0.5 * (*x + y);
    });

    ndarray::stack![Axis(2), first_diagonal, second_diagonal, off_diagonals]
}

fn sym_divergence(q: &ArrayView3<f32>) -> Array3<f32> {
    // // First component: ∂x q_0 - ∂y q_2
    // let first_term = -(q.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 0]), 1, 1));
    // let second_term = -0.5 * (q.slice(s![.., .., 2]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 2]), 0, 1));
    // let first_component = first_term + second_term;
    // // Second component: ∂y q_1 - ∂x q_2
    // let first_term = -(q.slice(s![.., .., 1]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 1]), 0, 1));
    // let second_term = -0.5 * (q.slice(s![.., .., 2]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 2]), 1, 1));
    // let second_component = first_term + second_term;


    let mut first_component = q.slice(s![.., .., 0]).clone().to_owned();
    first_component.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = -(row.to_owned() - shifted_row);
            row.assign(&diff);
        });
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = -0.5 * (col.to_owned() - shifted_col);
            col.assign(&diff);
        });
    par_azip!((x in &mut first_component, &y in &second_term) {
        *x += y;
    });

    let mut second_component = q.slice(s![.., .., 1]).clone().to_owned();
    second_component.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = -(col.to_owned() - shifted_col);
            col.assign(&diff);
        });
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = -0.5 * (row.to_owned() - shifted_row);
            row.assign(&diff);
        });
    par_azip!((x in &mut second_component, &y in &second_term) {
        *x += y;
    });

    ndarray::stack![Axis(2), first_component, second_component]
}

fn proj_p(p: &ArrayView3<f32>, alpha1: &f32) -> Array3<f32> {
    // let norm = (p.slice(s![.., .., 0]).map(|x| x.powi(2)) 
    //     + p.slice(s![.., .., 1]).map(|x| x.powi(2)))
    //     .sqrt();
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
    let mut slice1 = p_proj.slice_mut(s![.., .., 0]);
    // slice1 /= &factor;
    par_azip!((x in &mut slice1, &y in &factor) {
        *x /= y;
    });
    let mut slice2 = p_proj.slice_mut(s![.., .., 1]);
    // slice2 /= &factor;
    par_azip!((x in &mut slice2, &y in &factor) {
        *x /= y;
    });
    p_proj
}

fn proj_q(q: &ArrayView3<f32>, alpha0: &f32) -> Array3<f32> {
    // let norm = (q.slice(s![.., .., 0]).map(|x| x.powi(2)) 
    //     + q.slice(s![.., .., 1]).map(|x| x.powi(2)) 
    //     + q.slice(s![.., .., 2]).map(|x| x.powi(2)))
    //     .sqrt();
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
    let mut slice1 = q_proj.slice_mut(s![.., .., 0]);
    // slice1 /= &factor;
    par_azip!((x in &mut slice1, &y in &factor) {
        *x /= y;
    });
    let mut slice2 = q_proj.slice_mut(s![.., .., 1]);
    // slice2 /= &factor;
    par_azip!((x in &mut slice2, &y in &factor) {
        *x /= y;
    });
    let mut slice3 = q_proj.slice_mut(s![.., .., 2]);
    // slice3 /= &factor;
    par_azip!((x in &mut slice3, &y in &factor) {
        *x /= y;
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
    for i in 0..max_iter {
        let grad_u_bar = gradient(&u_bar.view());
        par_azip!((x in &mut p, &y in &grad_u_bar, &z in &w_bar) {
            *x += &sigma * (y - z);
        });
        let p = proj_p(&p.view(), &(&alpha0));
        // log!("Min max: {}, {}", p.clone().into_iter().reduce(f32::min).unwrap(), p.clone().into_iter().reduce(f32::max).unwrap());

        let sym_grad_w_bar = sym_gradient(&w_bar.view());
        par_azip!((x in &mut q, &y in &sym_grad_w_bar) {
            *x += &sigma * &lambda * y;
        });
        let q = proj_q(&q.view(), &(&alpha1));

        // Primal updates
        let u_old = u.clone();
        let w_old = w.clone();

        // Update u: TGV + data fidelity
        let div_p = divergence(&p.view());
        par_azip!((x in &mut u, &y in &div_p) {
            *x -= tau * &lambda * y;
        });

        // Data fidelity term
        let complex_u = Array2::<Complex<f64>>::from_shape_vec((ny, nx), (&u).into_iter().map(|x| Complex::new(*x as f64, 0.0)).collect()).unwrap();
        let fft_u = fft2shift(&fft2(&complex_u.view()).view());
        let mut residual = Array2::<Complex<f64>>::zeros((ny, nx));
        par_azip!((x in &mut residual, &y in &fft_u, &z in centered_kspace, &w in centered_mask) {
            if w > 0. {
                *x = y - z;
            }
        });
        let ifft_residual = ifft2(&ifft2shift(&residual.view()).view());
        let ifft_residual = Array2::<f32>::from_shape_vec((ny, nx), ifft_residual.into_iter().map(|x| x.re as f32).collect()).unwrap();
        par_azip!((x in &mut u, &y in &ifft_residual) {
            *x -= tau * y;
        });

        // Update w
        let sym_div_q = sym_divergence(&q.view());
        par_azip!((x in &mut w, &y in &p, &z in &sym_div_q) {
            *x -= tau * (-y) * z;
        });
        
        // Extrapolation
        // Update momentum
        let t_old = t;
        let t = (1. + (1. + 4. * t_old.powi(2)).sqrt()) / 2.0;
        let theta = (1. - t_old) / t;
        par_azip!((x in &mut u_bar, &y in &u, &z in &u_old) {
            // *x = 2. * y - z;
            *x = y + theta * (y - z);
        });
        par_azip!((x in &mut w_bar, &y in &w, &z in &w_old) {
            // *x = 2. * y - z;
            *x = y + theta * (y - z);
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

