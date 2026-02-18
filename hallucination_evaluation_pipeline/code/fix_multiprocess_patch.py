
import sys
import multiprocess.resource_tracker

def apply_patch():
    if sys.version_info >= (3, 12):
        # Monkeypatch ResourceTracker._stop to ignore AttributeError regarding _recursion_count
        # This error is caused by Python 3.12 and can be safely ignored.
        original_stop = multiprocess.resource_tracker.ResourceTracker._stop

        def _stop_patched(self):
            try:
                original_stop(self)
            except AttributeError as e:
                if "'_thread.RLock' object has no attribute '_recursion_count'" in str(e):
                    pass
                else:
                    raise e
            except Exception:
                 # Re-raise other exceptions
                raise

        multiprocess.resource_tracker.ResourceTracker._stop = _stop_patched
