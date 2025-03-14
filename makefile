ARCHIVE = FIA-PL3-AlexandreFonseca-DavidCarvalheiro-LuísGóis.zip
REPORT  = docs/relatorio.pdf

PANDOC_OPTS += --resource-path=docs
PANDOC_OPTS += --filter=pandoc-include

PYTHON_SCRIPTS := $(wildcard src/*.py)

%.pdf: %.md $(PYTHON_SCRIPTS)
	pandoc $(PANDOC_OPTS) $< --output=$@

.PHONY: archive
archive: $(ARCHIVE)

$(ARCHIVE): $(REPORT) $(PYTHON_SCRIPTS)
	rm --force $@
	zip $@ $^
