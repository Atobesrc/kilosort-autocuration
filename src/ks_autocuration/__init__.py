"""ks-autocuration: automated post-Kilosort curation that re-exports Kilosort format."""
from .pipeline import run

__version__ = "0.1.0"
__all__ = ["run", "__version__"]
