%.pdf: %.md
	pandoc --from=markdown-implicit_figures $< --output=$@
