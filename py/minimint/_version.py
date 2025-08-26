from importlib.metadata import version

try:
    __version__ = version("minimint")
except Exception:
    # Fallback for development installs
    __version__ = "unknown"

# For backward compatibility
version = __version__
