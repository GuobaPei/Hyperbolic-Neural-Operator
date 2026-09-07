# Paper source

The author-provided paper is stored as ordered binary parts so that each upload stays small. The Pages workflow concatenates them into `docs/assets/paper.pdf` without changing its contents.

To replace the paper, run `python scripts/prepare_paper.py /path/to/paper.pdf`, update the corresponding abstract, extracted text and metadata, and commit the resulting parts.
