#!/usr/bin/env python3
"""Download what the CourseLabs image serves under /vendor/ and /models/ into
./web: three npm packages and two ONNX models, about 315 MB. Standard library only.
Then run:  NLPLAB_WEB=web python3 serve.py   and open http://localhost:8013/
"""
import io, os, tarfile, urllib.request

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
NPM = (("@huggingface/transformers", "4.3.0"), ("onnxruntime-web", "1.31.0-dev.20260914-8d85527a0"),
       ("@huggingface/jinja", "0.5.10"), ("@huggingface/tokenizers", "0.2.0"),
       ("@sqlite.org/sqlite-wasm", "3.53.4-build1"))
MODELS = (("Xenova/all-MiniLM-L6-v2", ["config.json", "tokenizer.json", "tokenizer_config.json",
                                       "special_tokens_map.json", "onnx/model_quantized.onnx"]),
          ("HuggingFaceTB/SmolLM2-135M-Instruct", ["config.json", "generation_config.json", "tokenizer.json",
                                                   "tokenizer_config.json", "special_tokens_map.json",
                                                   "onnx/model_quantized.onnx"]))

def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()

for pkg, ver in NPM:
    base = pkg.split("/")[-1]
    t = tarfile.open(fileobj=io.BytesIO(get(f"https://registry.npmjs.org/{pkg}/-/{base}-{ver}.tgz")))
    dest = os.path.join(W, "node_modules", pkg)
    for m in t.getmembers():
        if m.isfile() and m.name.startswith("package/"):
            m.name = m.name[len("package/"):]
            t.extract(m, dest)
    print("npm", pkg, ver)
for repo, files in MODELS:
    for f in files:
        path = os.path.join(W, "models", repo, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            out.write(get(f"https://huggingface.co/{repo}/resolve/main/{f}"))
    print("model", repo)
print("ready: NLPLAB_WEB=web python3 serve.py")
