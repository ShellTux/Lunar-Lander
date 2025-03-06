PANDOC_OPTS += --resource-path=docs
PANDOC_OPTS += --filter=pandoc-include

PYTHON_SCRIPTS := $(wildcard src/*.py)

%.pdf: %.md $(PYTHON_SCRIPTS)
	pandoc $(PANDOC_OPTS) $< --output=$@
