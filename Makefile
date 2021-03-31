.PHONY: data clean env_create env_export env_activate env_build test create_kernel

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = demographic-modeling-module
ENV_NAME = demogrphic-modeling
CONDA_PARENT = arcgispro-py3

# make conda commands available in the script, notably for activation
# CONDA_BASE=$(conda info --base)
# source $CONDA_BASE/etc/profile.d/conda.sh

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data:
	. activate $(ENV_NAME)
	python src/data/make_dataset.py
	@echo ">>> Data processed."

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Build the local environment from the environment file
env:
	conda env create -f environment.yml
	@echo ">>> New conda environment, $(ENV_NAME), created. Activate with:\n- source activate $(ENV_NAME)\n- make env_activate"

create_kernel:
	python -m ipykernel install --user --name $(ENV_NAME) --display-name "$(PROJECT_NAME)"

docs:
    (cd ./docsrc && make html)
    cp -r ./docsrc/build/html/* ./docs

## Run all tests in module
test:
	. activate $(ENV_NAME)
	pytest

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
