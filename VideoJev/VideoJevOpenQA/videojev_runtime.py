"""Runtime helpers for running the released VideoJev HF model on this machine.

Kept identical in behaviour to the fixes in scripts/ground_videosiah.py:
  * decode-time frame cap: the official reader holds every sampled frame at FULL resolution
    before resizing to a few hundred px (a 4K video is ~47 GB per process -> OOM-killed), and
    cannot stack videos whose resolution changes mid-stream. Frames are resized at decode time
    to the first frame's size with the long edge <= cap; videos within the cap decode
    bit-identically to the official path.
  * shrink-on-OOM: 24 GB GPUs cannot hold the official contexts, so retry at 2/3 the context.
"""
import importlib
import inspect


def videojev_modules(model):
    package = type(model).__module__.rsplit('.', 1)[0]
    return {name: importlib.import_module(f'{package}.{name}')
            for name in ('modeling_videojev', 'processing_videojev', 'videojev_video', 'videojev_temporal')}


def patch_decode_cap(model, cap):
    """Install the capped reader where prepare_video looks it up. cap <= 0 leaves it official."""
    if not cap or cap <= 0:
        return
    modules = videojev_modules(model)
    video_module = modules['videojev_video']
    source = inspect.getsource(video_module.read_video)
    if source.count(".to_ndarray(format='rgb24')") != 2:
        raise RuntimeError('read_video changed upstream; refusing to patch frame decoding blindly')
    target = {}

    def capped_rgb(frame):
        if 'size' not in target:
            scale = min(1., cap / max(frame.width, frame.height))
            target['size'] = (max(2, round(frame.width * scale / 2) * 2) if scale < 1 else frame.width,
                              max(2, round(frame.height * scale / 2) * 2) if scale < 1 else frame.height)
        width, height = target['size']
        if (frame.width, frame.height) == (width, height):
            return frame.to_ndarray(format='rgb24')
        return frame.to_ndarray(format='rgb24', width=width, height=height)

    namespace = dict(vars(video_module), _capped_rgb=capped_rgb)
    exec(compile(source.replace("chosen.to_ndarray(format='rgb24')", '_capped_rgb(chosen)')
                 .replace("previous.to_ndarray(format='rgb24')", '_capped_rgb(previous)'),
                 video_module.__file__, 'exec'), namespace)
    patched = namespace['read_video']

    def read_video(*args, **kwargs):
        target.clear()
        return patched(*args, **kwargs)
    namespace['read_video'] = read_video
    modules['processing_videojev'].read_video = read_video


def run_with_context_fallback(fn, context, min_context, log=print):
    """fn(context) -> result; on CUDA OOM retry at 2/3 the context (64-aligned) down to min_context."""
    import torch
    while True:
        try:
            return fn(context), context
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            smaller = (context * 2 // 3 + 63) // 64 * 64
            if smaller < min_context:
                raise
            log(f'OOM at context={context}, retrying at {smaller}')
            context = smaller


def make_prepare_video(model, max_pixels=640 * 640, keep=None):
    """The released prepare_video with its per-frame pixel ceiling (hard-coded 640*640) set to
    max_pixels, e.g. 224*224 to match LongVT's max_pixels=50176. The context budget still applies;
    the resolution is min(context-limited, max_pixels). If `keep` is a dict, the decoded sample
    (raw frames + times) is stored in keep['sample'] so callers can re-encode a subset of the SAME
    frames at another resolution. Call after patch_decode_cap so the capped reader is used."""
    processing = videojev_modules(model)['processing_videojev']
    if max_pixels == 640 * 640 and keep is None:
        return processing.prepare_video
    source = inspect.getsource(processing.prepare_video)
    edits = [('640*640', '_max_pixels'),
             ("    n, height, width, _ = sample['video'].shape\n",
              "    n, height, width, _ = sample['video'].shape\n    if _keep is not None: _keep['sample'] = sample\n")]
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f'prepare_video changed upstream; cannot patch {old!r}')
        source = source.replace(old, new)
    namespace = dict(vars(processing), _max_pixels=int(max_pixels), _keep=keep)
    exec(compile(source, processing.__file__, 'exec'), namespace)
    return namespace['prepare_video']


def encode_frames(model, processor, frames, max_pixels):
    """Native VideoJev/Qwen3.5 merger features of `frames` (uint8 [n,H,W,3], n even) with each frame
    resized to at most max_pixels. Returns (features [tokens, dim], [t, h, w] grid in 16 px patches)."""
    import torch
    from transformers.video_utils import VideoMetadata
    vp = processor.video_processor
    n = len(frames)
    meta = VideoMetadata(total_num_frames=n, fps=1.0, frames_indices=list(range(n)))
    native = vp(videos=[frames], video_metadata=[meta], do_sample_frames=False,
                size=dict(shortest_edge=vp.size['shortest_edge'], longest_edge=n * int(max_pixels)),
                return_tensors='pt')
    grid = native['video_grid_thw']
    features = model.encode_video(native['pixel_values_videos'], grid)
    return features, grid[0].tolist()
