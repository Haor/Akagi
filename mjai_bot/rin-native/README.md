# RIN Native for Akagi 3.7.1

This four-player bot runs the RIN semantic actor locally with PyTorch. The
default model ID is `ppo400`, using CUDA device 0, FP32, SDPA attention, four
CPU threads, and disabled TF32. A resident loopback service loads the model
once and maintains a separate game state for each live game or review session.
Missing devices fail explicitly; they do not silently fall back to CPU.

## Local assets

Install a self-contained native Python runtime and a model bundle separately.
The source repository does not distribute model weights or a Python runtime.

```text
rin-native/
  bot.py
  manifest.toml
  src/
  runtime/python/python.exe
  models/ppo400/
    actor.safetensors
    config.json
    conversion-manifest.json
```

The runtime must already contain PyTorch, NumPy, safetensors, msgpack, filelock,
and a compatible Libriichi extension exposing `PlayerState.public_shanten()`.
The installer verifies these requirements without downloading packages. Keep
the runtime's third-party license files with it. Windows CUDA is the primary
deployment target; choosing `cpu` or `mps` does not imply separate platform
acceptance.

From this directory, install supplied assets with:

```powershell
python scripts/install_assets.py --runtime <native-python-directory> --model-bundle <converted-model-directory>
```

Existing destinations are preserved. For an externally managed runtime, set
`RIN_NATIVE_PYTHON` to its Python executable, or `RIN_NATIVE_RUNTIME` to its
directory. The default plugin-local runtime moves together with the bot.
In Akagi's Bots page, install the launcher's environment, then enable
**RIN Native (PPO 400)**. Akagi requires a separate launcher environment;
its `pyproject.toml` declares no dependencies because the launcher uses only
the standard library. Inference runs in the separately installed native
runtime, so creating the launcher environment needs no third-party packages.

## Generic model conversion

Supported checkpoints use the versioned `rin.semantic-dynamic.v1` architecture
and the public semantic input contract. Convert compatible future checkpoints
through the same command, with a new model ID:

```powershell
runtime/python/python.exe scripts/export_actor.py --source <checkpoint.msgpack> --config <config.json> --params-key params --output models/<model-id>
```

`--params-key` accepts a slash-separated subtree; use an empty value for a
parameters-only checkpoint. Conversion validates the architecture, tensor
shapes, finite FP32 values, per-tensor hashes, and the saved round trip. The
manifest binds the source checkpoint, config, and safetensors weights. There
are no model-name or hash allowlists. Set **Model ID** in Akagi after installing
the resulting bundle. Unsupported architectures require an explicit versioned
adapter.

## Public input and review contract

Open Akagi's **Game Review** page and choose **Local RIN** to review a complete
four-player history recording. No external API key is required. Results are
saved locally and show the recorded move, recommendation, and candidate
probabilities alongside a synchronized table. Step through recorded events,
play them at an adjustable speed, or jump to a round or decision. The table
uses Akagi's live-game renderer with a separate replay tracker; opponents'
hands stay hidden. Chinese locales display tile names in Chinese.
The live bot keeps its own independent session.
Use **Cloud Review** only when you intend to submit a recording to the configured
external service.

This adapter uses `rin.public-history.complete-round.v1` and
`rin.shanten.pre-draw-or-post-call.v1`, matching the PPO 400 semantic input
contract. Libriichi determines legal moves. The actor sees public MJAI history
and the selected player's hand; opponent starting hands and draws are censored
before updating either state representation. End-of-round tails remain in
history even if they did not lead to another decision.

Recommendations never update game state. A proposed riichi discard is retained
only when the next actual relevant event declares that player's riichi. This
allows historical games to follow their recorded actions even when RIN would
have chosen differently.

Every legal decision, including a pass, returns `meta.decision = true`, a
path-free `model_identity`, and atomic `candidates`. Each candidate contains an
MJAI `action`, normalized atomic `probability`, summed `protocol_probability`,
`protocol_class`, and `selected`. Riichi candidates also contain their intended
discard in `continuation`. The subsequent forced discard is marked
`meta.continuation = true`. `meta.show` provides Akagi's structured candidate
card. Probabilities are policy preferences, not win probabilities.

The Game page also includes an observation-head tile, which can be moved or
hidden like other tiles. Actor-only bundles report
`meta.observer.status = "unavailable"` and `reason = "no_compatible_observer"`,
bound to the current actor hash. The tile displays this status; it does not
substitute predictions from unrelated observation-head weights.

The service is shared only when settings, model bytes, Python identity,
Libriichi binary, and RIN source bytes match. Endpoint tokens and logs live in
the local `.runtime/` directory. To inspect or stop services, use:

```powershell
$env:PYTHONPATH = "$PWD/src"
runtime/python/python.exe -m rin.inference.native.client --root . --action check
runtime/python/python.exe -m rin.inference.native.client --root . --action stop-all
```

## Verification and source compatibility

```powershell
runtime/python/python.exe -m unittest discover -s tests -v
```

Behavior checks cover hidden-state masking, full round history, counterfactual
riichi, forced riichi continuation, and pass metadata. Numerical actor parity
and application acceptance are separate from these protocol tests.

`scripts/extract_public_adapter.py <source-root>` rebuilds the deployment
adapter from a compatible RIN source tree. It retains the public action codec,
semantic state, history, shanten, and protocol-equivalence policy while
excluding JAX/Flax actor loading. The native runtime does not import JAX,
Flax, or MahJax.

RIN actor and public-adapter implementation: Haor's RIN project. Akagi remains
attributed to its upstream authors under the repository's license and NOTICE.
Libriichi, PyTorch, and other separately installed runtime components retain
their respective licenses; no runtime binaries are vendored here.
