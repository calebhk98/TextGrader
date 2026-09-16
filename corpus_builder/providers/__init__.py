from .gutenberg import GENRE_TERMS, GutenbergProvider
from .public import GoogleBooksProvider, InternetArchiveProvider, LibraryOfCongressProvider, StandardEbooksProvider, WikisourceProvider

PROVIDER_TYPES = {
    "gutenberg": GutenbergProvider,
    "standard_ebooks": StandardEbooksProvider,
    "internet_archive": InternetArchiveProvider,
    "wikisource": WikisourceProvider,
    "google_books": GoogleBooksProvider,
    "library_of_congress": LibraryOfCongressProvider,
}

__all__ = ["GENRE_TERMS", "PROVIDER_TYPES", *[kind.__name__ for kind in PROVIDER_TYPES.values()]]
