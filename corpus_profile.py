"""Compatibility entry point for the corpus-profile builder."""
from textgrader.corpus import build_profile, load_profile, main, write_profile

__all__ = ["build_profile", "load_profile", "write_profile", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
