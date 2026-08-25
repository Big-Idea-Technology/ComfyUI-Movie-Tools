
import itertools
import logging
import os
import shutil
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import folder_paths
import json
import torch
from comfy.cli_args import args
from comfy.utils import common_upscale
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps

VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')
X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]


def _list_output_subfolders():
    """All subfolders of the ComfyUI output directory, relative to it, so the
    source folder can be picked from a dropdown instead of typed by hand."""
    output_dir = folder_paths.get_output_directory()
    folders = ["."]
    for root, dirs, _ in os.walk(output_dir):
        dirs.sort()
        for d in dirs:
            folders.append(os.path.relpath(os.path.join(root, d), output_dir))
    return folders


def _resolve_directory(directory):
    """Relative paths are resolved against the ComfyUI output directory;
    absolute paths are used as-is."""
    if os.path.isabs(directory):
        return directory
    return os.path.join(folder_paths.get_output_directory(), directory)


def _collect_image_paths(directory, image_load_cap=0, start_index=0):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

    paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(VALID_IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, file))

    if not paths:
        raise FileNotFoundError(f"No files in directory '{directory}'.")

    paths.sort()
    paths = paths[start_index:]
    if image_load_cap > 0:
        paths = paths[:image_load_cap]
    return paths


def _bounded_ordered_map(fn, items, max_workers, window):
    """Runs fn over items on a thread pool, yielding results in order while
    keeping at most `window` results decoded/in-flight at once, so memory use
    does not grow with the number of items (important for long sequences)."""
    max_workers = max(1, max_workers)
    window = max(max_workers, window)
    items = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = deque(ex.submit(fn, item) for item in itertools.islice(items, window))
        for item in items:
            yield futures.popleft().result()
            futures.append(ex.submit(fn, item))
        while futures:
            yield futures.popleft().result()


def _get_ffmpeg_path():
    env_override = os.environ.get("MOVIE_TOOLS_FFMPEG_PATH")
    if env_override:
        return env_override
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and make sure it is on PATH, "
            "or `pip install imageio-ffmpeg`."
        )


class SaveImagesWithSubfolder:

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "subfolder_name": ("STRING", {"default": "", "tooltip": "Create subfolder for this images."})
            },
            "optional": {
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1, "tooltip": "PNG compression level. Lower is faster but produces larger files, higher is slower but smaller. 0-1 is recommended for large batches."}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def run(self, images, filename_prefix="ComfyUI", subfolder_name="", compress_level=4, prompt=None, extra_pnginfo=None):
        filename_prefix = filename_prefix.replace("%subfolder_name%", str(subfolder_name))
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        results = list()
        tasks = list()
        for batch_number, image in enumerate(images):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            tasks.append((image, os.path.join(full_output_folder, file)))
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        def _save(task):
            image, full_path = task
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img.save(full_path, pnginfo=metadata, compress_level=compress_level)

        # PIL releases the GIL during PNG compression, so threads give a real
        # speedup here even though Python itself is single-threaded.
        max_workers = min(8, os.cpu_count() or 4, len(tasks))
        if max_workers <= 1:
            for task in tasks:
                _save(task)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                # list() forces iteration so exceptions from worker threads propagate.
                list(ex.map(_save, tasks))

        return { "ui": { "images": results } }


class LoadImagesFromSubdirsBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": -1, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "run"

    CATEGORY = "image"
    DESCRIPTION = "Loads images from a directory (and subdirectories) into a single batch tensor. For very long sequences (hundreds+ of frames) prefer 'Load images and create video', which streams frames straight to a video file instead of holding the whole batch in memory."

    def run(self, directory: str, image_load_cap: int = 0, start_index: int = 0):
        file_paths = _collect_image_paths(directory, image_load_cap, start_index)

        if len(file_paths) > 300:
            logging.warning(
                f"[Movie Tools] Loading {len(file_paths)} images into a single batch tensor. "
                "This can use a lot of RAM/VRAM and may OOM; consider 'Load images and create "
                "video' for long sequences instead."
            )

        def _load(path):
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            return torch.from_numpy(image)[None,]

        # Decoding releases the GIL, so parallelizing disk reads meaningfully
        # speeds up loading large directories.
        workers = min(8, os.cpu_count() or 4, len(file_paths))
        if workers <= 1:
            images = [_load(p) for p in file_paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                images = list(ex.map(_load, file_paths))

        if len(images) == 1:
            return (images[0], )

        image1 = images[0]
        batch = [image1]
        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear", "center").movedim(1, -1)
            batch.append(image2)

        # A single concat at the end avoids the O(n^2) cost of repeatedly
        # concatenating onto a growing tensor inside the loop.
        return (torch.cat(batch, dim=0), )


class LoadImagesAndCreateVideo:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": (_list_output_subfolders(), {"tooltip": "Folder with the image sequence, relative to the ComfyUI output directory (recurses into subdirectories)."}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "filename": ("STRING", {"default": "video", "tooltip": "Output file name in the ComfyUI output directory. May include a subfolder, e.g. 'my_folder/video'. '.mp4' is appended if missing. Existing files are overwritten."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio to mux into the video. Trimmed/padded to match the video length."}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "step": 1}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8, "tooltip": "Downscale so the longest side does not exceed this many pixels (0 = keep original resolution). Smaller frames encode much faster."}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1, "tooltip": "x264 quality. Lower is higher quality/larger file."}),
                "preset": (X264_PRESETS, {"default": "veryfast", "tooltip": "x264 speed/efficiency tradeoff. Faster presets encode quicker at a slightly larger file size."}),
                "num_workers": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1, "tooltip": "Parallel image decode threads. 0 = auto (CPU count, capped at 8)."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "run"

    OUTPUT_NODE = True

    CATEGORY = "image/video"
    DESCRIPTION = (
        "Loads an image sequence and encodes it straight to a video file (optionally with audio) "
        "without ever holding the whole sequence in memory. Frames are decoded on a bounded thread "
        "pool and streamed directly into ffmpeg, so sequences of thousands of frames won't OOM."
    )

    def run(self, directory, frame_rate=24.0, filename="video",
            audio=None, image_load_cap=0, start_index=0, max_side=0, crf=19, preset="veryfast",
            num_workers=0):
        file_paths = _collect_image_paths(_resolve_directory(directory), image_load_cap, start_index)
        frame_count = len(file_paths)

        with Image.open(file_paths[0]) as first_image:
            width, height = ImageOps.exif_transpose(first_image).convert("RGB").size

        if max_side > 0 and max(width, height) > max_side:
            scale = max_side / float(max(width, height))
            width, height = round(width * scale), round(height * scale)
        # yuv420p requires even dimensions.
        width, height = max(2, width - width % 2), max(2, height - height % 2)
        target_size = (width, height)

        output_dir = folder_paths.get_output_directory()
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
        video_only_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(video_only_path) or output_dir, exist_ok=True)
        subfolder = os.path.relpath(os.path.dirname(video_only_path), output_dir)
        if subfolder == ".":
            subfolder = ""
        final_path = video_only_path

        ffmpeg_bin = _get_ffmpeg_path()
        workers = num_workers if num_workers > 0 else min(8, os.cpu_count() or 4)

        def _load_frame(path):
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                if img.size != target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                return np.asarray(img, dtype=np.uint8)

        video_args = [
            ffmpeg_bin, "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(frame_rate), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", video_only_path,
        ]

        proc = subprocess.Popen(video_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            # Decoding runs on a bounded worker pool so at most a handful of
            # frames are ever resident in memory, regardless of how many
            # thousands of images are in the sequence.
            for frame in _bounded_ordered_map(_load_frame, file_paths, workers, window=workers * 3):
                proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass
        finally:
            proc.stdin.close()
            ret = proc.wait()

        if ret != 0:
            stderr = proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"ffmpeg failed while encoding frames (exit {ret}):\n{stderr}")

        if audio is not None and audio.get("waveform") is not None:
            # ffmpeg can't read and write the same file, so mux to a sibling
            # temp file and atomically replace the video-only one.
            mux_path = video_only_path + ".mux.mp4"
            waveform = audio["waveform"].squeeze(0).transpose(0, 1).contiguous().numpy()
            channels = waveform.shape[1] if waveform.ndim > 1 else 1
            sample_rate = audio["sample_rate"]
            video_duration = frame_count / float(frame_rate)

            mux_args = [
                ffmpeg_bin, "-y", "-v", "error",
                "-i", video_only_path,
                "-ar", str(sample_rate), "-ac", str(channels), "-f", "f32le", "-i", "-",
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", f"apad=whole_dur={video_duration}",
                "-shortest", mux_path,
            ]
            res = subprocess.run(mux_args, input=waveform.tobytes(), capture_output=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg failed while muxing audio (exit {res.returncode}):\n{res.stderr.decode('utf-8', 'replace')}")
            os.replace(mux_path, video_only_path)

        preview = {
            "filename": os.path.basename(final_path),
            "subfolder": subfolder,
            "type": self.type,
            "format": "video/h264-mp4",
            "frame_rate": frame_rate,
            "fullpath": final_path,
        }
        return {"ui": {"gifs": [preview]}}


NODE_CLASS_MAPPINGS = {
    "SaveImagesWithSubfolder": SaveImagesWithSubfolder,
    "LoadImagesFromSubdirsBatch": LoadImagesFromSubdirsBatch,
    "LoadImagesAndCreateVideo": LoadImagesAndCreateVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagesWithSubfolder": "[Movie Tools] Save images",
    "LoadImagesFromSubdirsBatch": "[Movie Tools] Load images from subdirs",
    "LoadImagesAndCreateVideo": "[Movie Tools] Load images and create video",
}
