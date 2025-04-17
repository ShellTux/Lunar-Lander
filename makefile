ARCHIVE = FIA-PL3-TP2-AlexandreFonseca-DavidCarvalheiro-LuísGóis.zip
REPORTS = docs/TP1/relatorio.pdf docs/TP2/relatorio.pdf

PANDOC_OPTS += --resource-path=docs
PANDOC_OPTS += --filter=pandoc-include

PYTHON_SCRIPTS := $(wildcard src/*.py)

%.pdf: %.md $(PYTHON_SCRIPTS)
	pandoc $(PANDOC_OPTS) $< --output=$@

.PHONY: archive
archive: $(ARCHIVE)

# $(ARCHIVE): $(REPORTS) $(PYTHON_SCRIPTS)
FIA-PL3-%-AlexandreFonseca-DavidCarvalheiro-LuísGóis.zip: docs/%/relatorio.pdf $(PYTHON_SCRIPTS)
	rm --force $@
	zip $@ $^
