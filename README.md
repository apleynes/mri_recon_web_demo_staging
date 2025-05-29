# MRI Reconstruction Web Demo

This is a web demo for sampling and reconstruction of MRI images written in Rust using Leptos and compiled to WebAssembly with Trunk.

Requires the latest trunk version: `>=v0.21.14` as this bundles an updated `wasm-opt` package that resolves wasm parsing errors.

The demo is deployed on Github Pages at https://apleynes.github.io/mri_recon_web_demo

Roadmap:
- [x] Zero-filled reconstruction
- [x] Compressed sensing reconstruction
- [x] Accelerate reconstruction with WebGPU and Web Workers
- [x] Add a "reconstruction options" section

## Manual deployment to Vercel

1. Install vercel CLI
2. `vercel login`
3. Set up vercel project
4. Build with trunk locally: `trunk build --release`
5. Build with vercel: `vercel build --prod`
6. Deploy: `vercel deploy --prod --prebuilt`

## Local Usage

```bash
trunk serve --release --open
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Dev notes

Zero-filled reconstruction is slower than original version. Need to investigate why.
