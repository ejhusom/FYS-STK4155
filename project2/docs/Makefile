# ===================================================================
# File:     Makefile for Markdown
# Author:   Erik Johannes Husom
# Created:  2019-06-19
# -------------------------------------------------------------------
# PREREQUISITES:
# - latex
# - (sudo apt-get install) pandoc
# - (sudo apt-get install) pandoc-citeproc
# - (pip3 install) pandoc-fignos
# - (pip3 install) pandoc-eqnos
# - (pip3 install) pandoc-tablenos
# USAGE:
# - Run `make pdf` to produce PDF from markdown-file.
# MARKDOWN USAGE: See https://pandoc.org/MANUAL.html
# 
#=================================================================
TEX			= pdflatex
DOC_NAME	= report
BIB		    = references.bib
PDF			= $(DOC_NAME).pdf
#FIG_DIR		= ./figs
#=================================================================
# TODO: Add support for making of figures as prerequisites

pdf: $(PDF)
$(PDF): *.md
	pandoc *.md -o $(PDF) \
		--filter=pandoc-fignos \
		--filter=pandoc-eqnos \
		--filter=pandoc-tablenos \
		--filter=pandoc-citeproc \
		--number-sections \
		--table-of-contents \
		--pdf-engine=$(TEX)

clean:
	@rm -rf $(DOC_NAME).pdf
	@echo "Cleanup complete!"
