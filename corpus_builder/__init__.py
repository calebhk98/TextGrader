"""Multi-provider, dependency-free corpus acquisition package."""
from .builder import BuildError, build, discover, provider_health
from .config import ConfigError, Settings, from_mapping, load, validate
from .models import Candidate, CorpusProvider, Document, ProviderStatus, SearchQuery

__all__ = ["BuildError", "Candidate", "ConfigError", "CorpusProvider", "Document", "ProviderStatus", "SearchQuery", "Settings", "build", "discover", "from_mapping", "load", "provider_health", "validate"]
