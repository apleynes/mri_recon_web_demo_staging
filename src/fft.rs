use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;
use ndrustfft::{self, R2cFftHandler, FftHandler};
use nshare::{self, IntoNdarray2, IntoNdarray3, IntoImageLuma};
use wgpu::*;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuComplex {
    real: f32,
    imag: f32,
}

impl From<Complex<f64>> for GpuComplex {
    fn from(c: Complex<f64>) -> Self {
        Self {
            real: c.re as f32,
            imag: c.im as f32,
        }
    }
}

impl From<GpuComplex> for Complex<f64> {
    fn from(c: GpuComplex) -> Self {
        Complex::new(c.real as f64, c.imag as f64)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FftParams {
    width: u32,
    height: u32,
    inverse: u32,
    _padding: u32,
}

pub struct GpuFftContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    fft_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
}

impl GpuFftContext {
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
            fft_pipeline: None,
            bind_group_layout: None,
        })
    }

    fn init_fft_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shader_source = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
            @group(0) @binding(1) var<uniform> params: FftParams;

            struct FftParams {
                width: u32,
                height: u32,
                inverse: u32,
                _padding: u32,
            }

            fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
                return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
            }

            fn complex_exp(theta: f32) -> vec2<f32> {
                return vec2<f32>(cos(theta), sin(theta));
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
                
                // Simple DFT implementation (not optimal but functional)
                var result = vec2<f32>(0.0, 0.0);
                let sign = select(1.0, -1.0, params.inverse == 1u);
                
                for (var k: u32 = 0u; k < width; k++) {
                    for (var l: u32 = 0u; l < height; l++) {
                        let src_idx = l * width + k;
                        let angle = sign * 2.0 * 3.14159265359 * (f32(idx.x * k) / f32(width) + f32(idx.y * l) / f32(height));
                        let twiddle = complex_exp(angle);
                        let value = data[src_idx];
                        result = result + complex_mul(value, twiddle);
                    }
                }
                
                if (params.inverse == 1u) {
                    result = result / f32(width * height);
                }
                
                data[linear_idx] = result;
            }
        "#;

        let shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("FFT Compute Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("FFT Bind Group Layout"),
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

        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("FFT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("FFT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.fft_pipeline = Some(pipeline);
        self.bind_group_layout = Some(bind_group_layout);
        Ok(())
    }

    pub async fn fft2_gpu(&mut self, input: &ArrayView2<'_, Complex<f64>>) -> Result<Array2<Complex<f64>>, Box<dyn std::error::Error>> {
        if self.fft_pipeline.is_none() {
            self.init_fft_pipeline()?;
        }

        let (height, width) = input.dim();
        let size = width * height;

        // Convert input to GPU format
        let gpu_data: Vec<GpuComplex> = input.iter().map(|&c| c.into()).collect();
        let data_bytes = bytemuck::cast_slice(&gpu_data);

        // Create buffers
        let buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("FFT Data Buffer"),
            contents: data_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let params = FftParams {
            width: width as u32,
            height: height as u32,
            inverse: 0,
            _padding: 0,
        };

        let params_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("FFT Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("FFT Bind Group"),
            layout: self.bind_group_layout.as_ref().unwrap(),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("FFT Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("FFT Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.fft_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_x = ((width as u32) + 15) / 16;
            let workgroup_y = ((height as u32) + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("FFT Staging Buffer"),
            size: data_bytes.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, data_bytes.len() as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(Maintain::Wait);
        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let result_data: &[GpuComplex] = bytemuck::cast_slice(&data);
        let output: Vec<Complex<f64>> = result_data.iter().map(|&c| c.into()).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(Array2::from_shape_vec((height, width), output)?)
    }

    pub async fn ifft2_gpu(&mut self, input: &ArrayView2<'_, Complex<f64>>) -> Result<Array2<Complex<f64>>, Box<dyn std::error::Error>> {
        if self.fft_pipeline.is_none() {
            self.init_fft_pipeline()?;
        }

        let (height, width) = input.dim();
        let size = width * height;

        // Convert input to GPU format
        let gpu_data: Vec<GpuComplex> = input.iter().map(|&c| c.into()).collect();
        let data_bytes = bytemuck::cast_slice(&gpu_data);

        // Create buffers
        let buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("IFFT Data Buffer"),
            contents: data_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let params = FftParams {
            width: width as u32,
            height: height as u32,
            inverse: 1,
            _padding: 0,
        };

        let params_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("IFFT Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("IFFT Bind Group"),
            layout: self.bind_group_layout.as_ref().unwrap(),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("IFFT Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("IFFT Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.fft_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_x = ((width as u32) + 15) / 16;
            let workgroup_y = ((height as u32) + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
        }

        // Read back results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("IFFT Staging Buffer"),
            size: data_bytes.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffer, 0, &staging_buffer, 0, data_bytes.len() as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(Maintain::Wait);
        receiver.await??;

        let data = buffer_slice.get_mapped_range();
        let result_data: &[GpuComplex] = bytemuck::cast_slice(&data);
        let output: Vec<Complex<f64>> = result_data.iter().map(|&c| c.into()).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(Array2::from_shape_vec((height, width), output)?)
    }
}

// CPU implementations (existing)
pub fn fft2(gray_img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = gray_img.dim();
    let mut fft_img = Array2::<Complex<f64>>::zeros((ny, nx));
    let mut fft_handler = FftHandler::new(nx);
    ndrustfft::ndfft_par(&gray_img, &mut fft_img, &mut fft_handler, 1);
    let mut fft_handler = FftHandler::new(ny);
    ndrustfft::ndfft_par(&fft_img.clone(), &mut fft_img, &mut fft_handler, 0);
    fft_img
}

pub fn ifft2(fft_img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = fft_img.dim();
    let mut ifft_img = Array2::<Complex<f64>>::zeros((ny, nx));
    let mut ifft_handler = FftHandler::new(ny);
    ndrustfft::ndifft_par(&fft_img, &mut ifft_img, &mut ifft_handler, 0);
    let mut ifft_handler = FftHandler::new(nx);
    ndrustfft::ndifft_par(&ifft_img.clone(), &mut ifft_img, &mut ifft_handler, 1);
    ifft_img
}

// High-level API with automatic GPU/CPU fallback
pub async fn fft2_auto(gray_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    match GpuFftContext::new().await {
        Ok(mut ctx) => {
            match ctx.fft2_gpu(gray_img).await {
                Ok(result) => {
                    leptos::logging::log!("Using GPU FFT");
                    result
                },
                Err(_) => {
                    leptos::logging::log!("GPU FFT failed, falling back to CPU");
                    fft2(gray_img)
                }
            }
        },
        Err(_) => {
            leptos::logging::log!("WebGPU not available, using CPU FFT");
            fft2(gray_img)
        }
    }
}

pub async fn ifft2_auto(fft_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    match GpuFftContext::new().await {
        Ok(mut ctx) => {
            match ctx.ifft2_gpu(fft_img).await {
                Ok(result) => {
                    leptos::logging::log!("Using GPU IFFT");
                    result
                },
                Err(_) => {
                    leptos::logging::log!("GPU IFFT failed, falling back to CPU");
                    ifft2(fft_img)
                }
            }
        },
        Err(_) => {
            leptos::logging::log!("WebGPU not available, using CPU IFFT");
            ifft2(fft_img)
        }
    }
}

pub fn fft2shift(img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = img.dim();
    let mut shifted = Array2::zeros((ny, nx));
    let half_ny = ny / 2;
    let half_nx = nx / 2;

    for i in 0..ny {
        for j in 0..nx {
            let new_i = (i + half_ny) % ny;
            let new_j = (j + half_nx) % nx;
            shifted[[new_i, new_j]] = img[[i, j]];
        }
    }
    shifted
}

pub fn ifft2shift(img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = img.dim();
    let mut shifted = Array2::zeros((ny, nx));
    let half_ny = ny / 2;
    let half_nx = nx / 2;

    for i in 0..ny {
        for j in 0..nx {
            let new_i = (i + half_ny) % ny;
            let new_j = (j + half_nx) % nx;
            shifted[[new_i, new_j]] = img[[i, j]];
        }
    }
    shifted
}

fn convert_to_image_and_save(img: &Array2<u8>, filename: &str) {
    let (ny, nx) = img.dim();
    let img: GrayImage = GrayImage::from_raw(nx as u32, ny as u32, img.clone().into_iter().collect()).unwrap();
    img.save(filename).unwrap();
}
