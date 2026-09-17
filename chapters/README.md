# chapters

Put your manuscript's chapter files here, one per file. TextGrader reads this
directory when a project report needs the whole book rather than the single
file you passed to `grade.py`.

Name them with a leading number so they sort and so chapter numbering can be
checked: `01_opening.md`, `02_the_letter.md`, ... `100_epilogue.md`. The number
may be any length.

Point somewhere else with `chapters_dir` in `config.json`. Nothing here is
required: `grade.py draft.md` grades the file you name and never needs this
directory.

Related:

- `python3 build_manuscript.py` assembles these into one manuscript and checks
  for numbering gaps, duplicates, and headings that disagree with filenames.
- `python3 -m textgrader.corpus chapters/ --comparison-unit chapter -o x.json`
  builds a corpus profile where one observation is a chapter, so chapter-scale
  metrics compare against chapters rather than against whole novels.
