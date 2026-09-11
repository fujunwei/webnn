# LiteRT in-renderer inference: Tensor I/O fast path

Owner: WebNN team
Status: Phase 1 implemented (compile-clean; awaiting perf validation)
Related feature flag: `mojom::features::kWebNNLiteRT`
Last updated: 2026-07-01

## 1. Background

When the LiteRT / TFLite backend is selected in the WebNN renderer path, the
service (`ContextImplLiteRt` / `TensorImplTflite`) is created inside the
renderer process via [`WebNNContextProviderInRenderer`](webnn_context_provider_in_renderer.h).
`ContextImplLiteRt` runs on a dedicated `base::ThreadPool` single‑thread task
runner (the *owning task runner*), while Blink's `MLTensor`/`MLContext` live on
the renderer main thread. Communication uses the standard `mojom::WebNN*`
interfaces even though both ends share the same address space.

Concretely, every `MLTensor.writeTensor()` / `MLTensor.readTensor()` call today
performs **two data copies** and pays the cost of Mojo message serialization,
including — for tensors larger than `mojo_base::BigBuffer::kMaxInlineBytes`
(64 KB) — one shared‑memory region allocation, mapping and unmapping per call.

## 2. Current data flow (per `dispatch`)

`writeTensor(input, srcArrayBuffer)`

1. Blink main thread: `src` → `mojo_base::BigBuffer` **or** data pipe write
   (`memcpy #1`; shared‑memory region allocation for large tensors).
2. Mojo serialization + cross‑sequence dispatch onto the owning task runner
   (no payload copy but per‑message overhead).
3. Owning task runner: `TensorImplTflite::WriteTensorImpl` → `ResourceTask`
   acquires exclusive lock → `WebNNContextImpl::ReadDataFromBigBufferOrDataPipe`
   (`memcpy #2`) into `BufferContent`.

`dispatch(graph, inputs, outputs)`

1. Blink → `context_remote_->Dispatch(token, inputs, outputs)`; payload is
   tokens only, no tensor data.
2. Owning task runner → `WebNNGraphImpl::RunDispatch` → `GraphImplLiteRt::DispatchImpl`.
3. `::litert::TensorBuffer::CreateFromHostMemory` on each `BufferContent`
   (zero copy at this step).
4. `ComputeResources::DoDispatch` runs on `base::ThreadPool` (compute).

`readTensor(output)` → mirror of `writeTensor`, also two copies.

## 3. Optimization goals

1. Remove the two unnecessary copies on `writeTensor`/`readTensor` for the
   in‑renderer path (they exist only because the Mojo boundary requires a
   serialized payload).
2. Skip Mojo message serialization and the `BigBuffer` shared‑memory dance
   entirely — everything is in the same process.
3. Preserve the observable Web API semantics, including the FIFO ordering
   between `writeTensor` / `dispatch` / `readTensor` guaranteed by pipe
   ordering today.

Non‑goals for phase 1:
- Bypassing Mojo for `Dispatch` (deferred to phase 2).
- Zero‑copy write/read (would require cross‑thread lock acquisition on
  `BufferContent`; deferred).
- Changing behavior of the GPU‑process (out‑of‑process) path.

## 4. Design (phase 1)

Introduce a thread‑safe C++ side channel between Blink `MLTensor` and the
in‑renderer backend that reuses the existing `ResourceTask` machinery on the
owning task runner and bypasses Mojo for the two tensor I/O interfaces.

### 4.1 Public interface

New header [`services/webnn/public/cpp/in_renderer_tensor_handle.h`](public/cpp/in_renderer_tensor_handle.h)
exposes an abstract, ref‑counted handle plus a small global registry keyed by
`blink::WebNNTensorToken`:

```cpp
class COMPONENT_EXPORT(WEBNN_IN_RENDERER_TENSOR) InRendererTensorHandle
    : public base::RefCountedThreadSafe<InRendererTensorHandle> {
 public:
  // Copies |src| synchronously on the caller's sequence into a heap buffer,
  // then schedules the write on the backend's owning task runner.
  virtual void WriteTensor(base::span<const uint8_t> src) = 0;

  // Schedules a read on the backend's owning task runner. |done| is invoked
  // on the caller's sequence with the copied bytes.
  using ReadCallback = base::OnceCallback<void(std::vector<uint8_t>)>;
  virtual void ReadTensor(ReadCallback done) = 0;
};

class COMPONENT_EXPORT(WEBNN_IN_RENDERER_TENSOR) InRendererTensorRegistry {
 public:
  static void Register(const blink::WebNNTensorToken&,
                       scoped_refptr<InRendererTensorHandle>);
  static void Unregister(const blink::WebNNTensorToken&);
  static scoped_refptr<InRendererTensorHandle> Get(
      const blink::WebNNTensorToken&);
};
```

Both live in a new `source_set("in_renderer_tensor")` under
`services/webnn/public/cpp/BUILD.gn` with minimal deps (`//base`,
`//third_party/blink/public/common/tokens:tokens_headers`). The service target
and the Blink module both depend on it.

### 4.2 Service side

- `TensorImplTflite::Create` (only when `WebNNContextImpl::is_in_renderer()`)
  constructs a concrete `InRendererTensorHandle` that owns
  `scoped_refptr<QueueableResourceState<BufferContent>>` and the owning task
  runner, and registers it against the tensor's token.
- `WriteTensor(span)` copies once into a heap `std::vector` on the caller
  sequence, then `PostTask`s a lambda onto the owning task runner. The lambda
  builds a `ResourceTask` with an exclusive lock on the buffer state and
  performs a single `memcpy` into `BufferContent::AsSpan()`. No `BigBuffer`,
  no Mojo serialization, no shared‑memory region.
- `ReadTensor(done)` `PostTask`s a lambda that builds a `ResourceTask` with a
  shared lock, copies `BufferContent::AsSpan()` into a `std::vector`, then
  `PostTask`s the callback with the vector back onto the caller sequence.
- Unregistration happens in the tflite handle's destructor (when both service
  and Blink release their refs) so late uses cannot see stale entries.

### 4.3 Blink side

- After `MLContext::createTensor()` succeeds *on the in‑renderer path* (i.e.,
  the request went through `in_process_context_provider_`), `MLContext`
  synchronously calls `InRendererTensorRegistry::Get(token)` and passes the
  resulting handle to the `MLTensor` constructor. The reply is guaranteed to
  arrive after `TensorImplTflite::Create` has registered the handle because
  the reply is serialized after the `CreateTensor` completion on the owning
  task runner.
- `MLTensor::WriteTensorImpl` and both variants of `ReadTensorImpl` check
  whether `in_renderer_handle_` is set; if so they take the fast path and
  never touch `remote_tensor_->WriteTensor()` / `->ReadTensor()`.

### 4.4 Ordering with `Dispatch`

Both the fast‑path `PostTask` and the still‑Mojo `context_remote_->Dispatch()`
message end up as tasks queued on the same owning task runner. Because both
originate from the same source sequence (Blink main thread) and both are
observed in program order on the source, they arrive in the same order on the
target runner (Mojo pipe writes to an in‑process peer post the receiver task
via the standard task runner API). Hence:

```
Blink main thread: writeTensor(A) ; dispatch() ; readTensor(A)
Owning runner:     [Write A]      ; [Dispatch]  ; [Read A]
```

is preserved without additional synchronization. This assumption is documented
alongside the fast path in `ml_tensor.cc`.

### 4.5 Lifetime & safety

- `InRendererTensorHandle` is `RefCountedThreadSafe`. Blink holds one ref,
  the service side (`TensorImplTflite`) holds one ref. The last release is
  triggered by whichever side lets go last; there is no dependency from
  Blink onto `TensorImplTflite`'s destruction sequence.
- The underlying `QueueableResourceState<BufferContent>` is likewise
  ref‑counted and safely outlives either side.
- The registry itself uses a `base::NoDestructor<base::Lock + flat_map>` so it
  is safe to touch from any thread.

## 5. Expected impact

For each `writeTensor` / `readTensor` on the in‑renderer LiteRT path we save:

| Cost                              | Before      | After |
|-----------------------------------|-------------|-------|
| Data copies                       | 2           | 2\*   |
| `BigBuffer` allocation            | 1 (>64 KB → shared mem region)   | 0     |
| Shared‑memory map/unmap syscalls  | up to 2 pairs                    | 0     |
| Mojo message serialize/deserialize| 1 per call                       | 0     |

\* One copy is unavoidable in phase 1 (script‑thread → heap buffer) because we
cannot access V8 ArrayBuffer storage from the compute sequence. The second
copy (heap buffer → `BufferContent`) can be eliminated in a future revision if
we allow the fast path to acquire the exclusive lock on the source sequence
directly.

For large tensors (> 64 KB, the majority of production workloads such as
LLM inference) the elimination of `BigBuffer`/shared‑memory setup and Mojo
serialization dominates and yields the primary latency reduction; small
tensors mostly benefit from the removed Mojo scheduling overhead.

## 6. Rollout & follow‑ups

- Phase 1 (this doc): `writeTensor` / `readTensor` fast path. Behind no
  runtime flag — activated automatically whenever a tensor was created via
  the in‑renderer provider path, so existing tests exercise both code paths.
- Phase 2: also bypass Mojo for `Dispatch`. Requires exposing a direct
  `WebNNContextImpl` weak pointer to Blink and posting a task equivalent to
  `WebNNContextImpl::Dispatch()` onto the owning task runner.
- Phase 3 (optional): zero‑copy write path via cross‑thread lock acquisition
  on `BufferContent`; small‑tensor synchronous inlining.

## 7. File map

| File                                                           | Change |
|----------------------------------------------------------------|--------|
| [`public/cpp/in_renderer_tensor_handle.h`](public/cpp/in_renderer_tensor_handle.h)   | New: abstract handle + registry |
| [`public/cpp/in_renderer_tensor_handle.cc`](public/cpp/in_renderer_tensor_handle.cc) | New: registry impl |
| [`public/cpp/BUILD.gn`](public/cpp/BUILD.gn)                                          | New target `in_renderer_tensor` |
| [`tflite/tensor_impl_tflite.h`](tflite/tensor_impl_tflite.h)                          | New helper `CreateInRendererHandle()` |
| [`tflite/tensor_impl_tflite.cc`](tflite/tensor_impl_tflite.cc)                        | Fast‑path handle impl; register at create, unregister at destroy |
| [`webnn_context_impl.h`](webnn_context_impl.h)                                        | Getter `is_context_provider_in_renderer()`; register handle after `CreateTensorImpl` |
| [`webnn_context_impl.cc`](webnn_context_impl.cc)                                      | Registration call |
| [`../../third_party/blink/renderer/modules/ml/ml_context.{h,cc}`](../../third_party/blink/renderer/modules/ml/ml_context.h) | Pass in‑renderer flag; look up handle on success |
| [`../../third_party/blink/renderer/modules/ml/webnn/ml_tensor.{h,cc}`](../../third_party/blink/renderer/modules/ml/webnn/ml_tensor.h) | Hold handle; fast path in `WriteTensorImpl` / `ReadTensorImpl` |

