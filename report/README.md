# Report

This folder contains the LaTeX report source for the refusal-ablation project.
It uses the course-provided `arxiv.sty` style and follows an IMRAD structure:

- Introduction
- Methods
- Results
- Discussion
- Conclusion

Build from this folder with:

```bash
latexmk -pdf main.tex
```

If `latexmk` is unavailable, use:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
