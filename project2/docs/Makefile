# ===================================================================
# File:     Makefile for Latex
# Author:   Erik Johannes Husom
# Created:  2019-06-19
# -------------------------------------------------------------------
# PREREQUISITES:
# - latex
# USAGE:
# Run `make latex` to produce PDF from latex-files.
#=================================================================
TEX			= pdflatex
DOC_NAME	= report
BIB		    = references.bib
PDF			= $(DOC_NAME).pdf
#FIG_DIR		= ./figs
#=================================================================
# TODO: Add support for making of figures as prerequisites

latex: $(PDF)
$(PDF): $(DOC_NAME).tex #$(FIG_DIR)/%.pdf $(FIG_DIR)%.png
	$(TEX) $(DOC_NAME)
	# Using .bib-file if it exists:
	if test -e $(BIB); then bibtex $(DOC_NAME); fi
	# Multiple compiles necessary for correct handling of references:
	$(TEX) $(DOC_NAME)
	$(TEX) $(DOC_NAME)
	@rm -rf *.aux *.bbl *.blg *.log *.out *.toc
	@echo "PDF-file created, and temporary files removed."

clean:
	@rm -rf *.aux *.bbl *.blg *.log *.out *.toc
	@rm -rf $(DOC_NAME).pdf
	@echo "Cleanup complete!"

