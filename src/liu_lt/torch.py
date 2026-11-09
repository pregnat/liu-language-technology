import gc
import torch

from pathlib import Path
from IPython.display import display, HTML


MODEL_FOLDER = Path("/courses/TDP030/models/")
MIN_GB_VRAM = 10
GLOBALS_TO_CLEAN = {"tokenizer", "model", "pipe", "it_pipe", "pt_pipe", "streamer"}


def check_for_model(*model_names):
    model_paths = []
    for model_name in model_names:
        model_path = MODEL_FOLDER / model_name
        if not model_path.exists():
            model_path = model_name
        model_paths.append(model_path)

    check_gpu_requirements()
    print(model_paths)

    return (*model_paths,)


def check_gpu_requirements():
    if not torch.cuda.is_available():
        display(
            HTML(
                """<div class="alert alert-danger">
❌ <strong>The models in this lab require a GPU.</strong>

Your computer either does not have a GPU, or CUDA is not installed properly.  <strong>You will not be able to run this lab on your own computer.</strong>  You can access the GPU-equipped lab rooms in B-huset – Åsgard, Egypten, Olympen – to work on this lab, preferably during the times we have booked one or more of these rooms for the course.
</div>"""
            )
        )
        return

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_properties(0).name

    if gpu_memory_gb < MIN_GB_VRAM:
        display(
            HTML(
                f"""<div class="alert alert-warning">
⚠️ The models in this lab require a <strong>GPU with at least {MIN_GB_VRAM} GB VRAM.</strong>

You are running on a {gpu_name} with {gpu_memory_gb:.1f} GB VRAM.  <strong>You might encounter out-of-memory issues.</strong>  If this happens, remember that you can access the GPU-equipped lab rooms in B-huset – Åsgard, Egypten, Olympen – to work on this lab, preferably during the times we have booked one or more of these rooms for the course.
</div>"""
            )
        )
        return

    display(
        HTML(
            f"""<div class="alert alert-success">Running on a {gpu_name} GPU with {gpu_memory_gb:.1f} GB VRAM.</div>"""
        )
    )


def free_gpu_memory(clean_vars=None):
    if not clean_vars:
        clean_vars = GLOBALS_TO_CLEAN

    for var_name in clean_vars:
        if var_name in globals():
            del globals()[var_name]

    gc.collect()
    torch.cuda.empty_cache()
