import inspect
import math
import traceback
import sys

from contextlib import contextmanager
from pathlib import Path
from IPython.display import display, HTML


COURSE_FOLDER = Path("/courses/729G86/labs/")


def find_file(filename):
    """Find a file so that we don't need to specify whether we're running
    locally or on the lab room computers."""
    p = Path(filename)
    if not p.exists():
        current_file = inspect.getfile(inspect.currentframe())
        if (q := Path(current_file).parent / p.name).exists():
            return q
        for r in COURSE_FOLDER.rglob(p.name):
            return r
    return p


def success(msg: str = "Checks have passed!"):
    display(HTML(f'<div class="alert alert-success"><strong>{msg}</strong></div>'))


def error(msg: str = "Some checks have failed!"):
    display(HTML(f'<div class="alert alert-danger"><strong>{msg}</strong></div>'))


def get_caller_line(stack=None):
    """Find first line in the current stack that's not from within this module."""
    if not stack:
        stack = reversed(traceback.extract_stack())

    for frame in stack:
        if frame.filename != __file__:
            return frame.line

    return "<unknown line>"


@contextmanager
def check_solutions():
    """Context manager that runs all assertions and reports failures."""
    failures = []

    class AssertRecorder:
        def __call__(self, condition, message=""):
            try:
                assert condition, message
            except AssertionError as e:
                failures.append((get_caller_line(), str(e)))

        def isclose(self, a, b, message=""):
            if not message:
                message = f"{a} ≠ {b}"
            self(math.isclose(a, b), message=message)

        def equals(self, a, b, message=""):
            if not message:
                message = f"{a} ≠ {b}"
            self(a == b, message=message)

    recorder = AssertRecorder()

    try:
        yield recorder
    except Exception as e:
        # For an exception raised by the calling code, we need to inspect the traceback
        # from the exception itself, otherwise we return a line from the context
        # manager's exception handling mechanism, rather than the notebook.
        _, _, exc_tb = sys.exc_info()
        tb_list = traceback.extract_tb(exc_tb)

        failures.append((get_caller_line(tb_list), f"Raised {type(e).__name__}: {e}"))

        # Re-raise exception to get full traceback
        raise (e)
    finally:
        if failures:
            for line, msg in failures:
                print(f"❌ {line}")
                print(f"    {msg}")
            error()
        else:
            success()
