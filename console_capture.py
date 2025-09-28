import sys
import io
from contextlib import contextmanager

class TeeOutput:
    """Capture stdout/stderr while still printing to console"""
    def __init__(self, stream, callback=None):
        self.stream = stream
        self.callback = callback
        self.buffer = []
        
    def write(self, data):
        # Write to original stream
        self.stream.write(data)
        self.stream.flush()
        
        # Capture the data
        if data.strip():  # Only capture non-empty lines
            self.buffer.append(data.rstrip())
            if self.callback:
                self.callback(data.rstrip())
    
    def flush(self):
        self.stream.flush()
    
    def get_captured(self):
        return '\n'.join(self.buffer)

@contextmanager
def capture_console(stdout_callback=None, stderr_callback=None):
    """Context manager to capture console output"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_capture = TeeOutput(old_stdout, stdout_callback)
    stderr_capture = TeeOutput(old_stderr, stderr_callback)
    
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture
    
    try:
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr