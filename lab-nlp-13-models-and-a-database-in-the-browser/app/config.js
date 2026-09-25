// config.js: where Transformers.js is allowed to fetch from.
//
// Out of the box, Transformers.js downloads models from the Hugging Face Hub
// and the ONNX Runtime WebAssembly files from a CDN (cdn.jsdelivr.net). Both
// are exactly what this app must never do, so three settings point it at the
// copies this session serves instead.
import { env } from '/vendor/@huggingface/transformers/dist/transformers.js';

// 1. Never ask the Hugging Face Hub for a model.
env.allowRemoteModels = false;

// 2. Look for models here instead: '/models/Xenova/all-MiniLM-L6-v2/' and so on,
//    served by serve.py from /opt/nlplab/web/models.
env.allowLocalModels = true;
env.localModelPath = '/models/';

// 3. Load ONNX Runtime's WebAssembly engine from this server, not from the CDN.
env.backends.onnx.wasm.wasmPaths = {
  mjs: '/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs',
  wasm: '/vendor/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm',
};

export { env };
