# WebAssembly Threading Debug Guide

## Problem Summary

The application works with multi-threading locally but fails on GitHub Pages due to missing HTTP headers required for `SharedArrayBuffer` and WebAssembly threading.

## Root Cause

GitHub Pages doesn't support custom HTTP headers, specifically:
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`

These headers are required for `SharedArrayBuffer` and WebAssembly threading to work.

## Local Testing & Debugging

### Method 1: Using the Custom Test Server

1. Build the project:
   ```bash
   trunk build --release
   ```

2. Run the test server with proper CORS headers:
   ```bash
   python3 test-server.py 8080
   ```

3. Open `http://localhost:8080` in your browser
4. Check the browser console - you should see "Multi-threading enabled"

### Method 2: Using the Debug Script

1. Add the debug script to your HTML or run it in the console:
   ```html
   <script src="debug-threading.js"></script>
   ```

2. Check the console output for threading capability analysis

### Method 3: Compare Local vs GitHub Pages

**Local Development (trunk serve):**
- ✅ SharedArrayBuffer available
- ✅ Cross-origin isolated
- ✅ Multi-threading works

**GitHub Pages:**
- ❌ SharedArrayBuffer unavailable
- ❌ Not cross-origin isolated
- ❌ Falls back to single-threaded mode

## Testing the Fix

### Before the Fix
1. Deploy to GitHub Pages
2. Open browser console
3. You should see the error: "WebAssembly.Memory object cannot be serialized"

### After the Fix
1. Build with updated configuration:
   ```bash
   trunk build --release
   ```

2. Deploy to GitHub Pages
3. Open browser console
4. You should see: "SharedArrayBuffer not available (likely due to missing CORS headers). Running in single-threaded mode."
5. No errors should occur, just graceful degradation

## Verification Checklist

- [ ] Local development server shows "Multi-threading enabled"
- [ ] Test server (test-server.py) shows "Multi-threading enabled"
- [ ] GitHub Pages shows graceful fallback message
- [ ] No console errors on GitHub Pages
- [ ] Application functions correctly in single-threaded mode

## Alternative Deployment Options

If multi-threading is critical, consider these alternatives to GitHub Pages:

1. **Netlify**: Supports `_headers` files
2. **Cloudflare Pages**: Supports `_headers` files
3. **Vercel**: Supports custom headers via `vercel.json`
4. **Firebase Hosting**: Supports headers via `firebase.json`

## Performance Impact

- **Multi-threaded**: Utilizes all CPU cores for FFT operations
- **Single-threaded**: Limited to one core, slower but still functional
- **Recommendation**: Display a notice to users about performance differences

## Code Changes Made

1. **Trunk.toml**: Added conditional thread pool initialization
2. **index.html**: Updated CSP headers
3. **test-server.py**: Created local testing server
4. **debug-threading.js**: Created debugging utility

## Future Considerations

Consider adding a UI indicator showing whether multi-threading is active:

```javascript
// In your Rust/WASM code
if (threadingEnabled) {
    display_message("⚡ Multi-threading enabled - optimal performance");
} else {
    display_message("🔄 Running in compatibility mode - slower but stable");
}
``` 