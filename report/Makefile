PDF = main.pdf

.PHONY: all clean

all: $(PDF)

$(PDF): main.tex references.bib sections/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
