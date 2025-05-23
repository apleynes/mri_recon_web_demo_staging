# MRI Reconstruction Web Demo

This is a web demo for sampling and reconstruction of MRI images written in Rust using Leptos and compiled to WebAssembly with Trunk.

Requires the latest trunk version: `>=v0.21.14` as this bundles an updated `wasm-opt` package that resolves wasm parsing errors.

The demo is deployed on Github Pages at https://apleynes.github.io/mri_recon_web_demo

Roadmap:
- [x] Zero-filled reconstruction
- [ ] Compressed sensing reconstruction
- [ ] Accelerate reconstruction with WebGPU and Web Workers
- [ ] Add a "reconstruction options" section

## Local Usage

```bash
trunk serve --release --open
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Dev notes

Zero-filled reconstruction is slower than original version. Need to investigate why.
