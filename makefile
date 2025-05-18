ARCHIVE = FIA-PL3-TP2-AlexandreFonseca-DavidCarvalheiro-LuísGóis.zip

PANDOC_OPTS += --resource-path=docs
PANDOC_OPTS += --filter=pandoc-include

PYTHON_SCRIPTS := $(shell find src/alexandre -type f -name "*.py")

%.pdf: %.md $(PYTHON_SCRIPTS)
	pandoc $(PANDOC_OPTS) $< --output=$@

.PHONY: archive
archive: $(ARCHIVE)

FIA-PL3-%-AlexandreFonseca-DavidCarvalheiro-LuísGóis.zip: docs/%/relatorio.pdf $(PYTHON_SCRIPTS)
	rm --force $@
	zip --junk-paths $@ $^
