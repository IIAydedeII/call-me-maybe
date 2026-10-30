# =========================================
# ANSI Escape Sequences
# =========================================

# Basic
ESC			:=	\033
RESET		:=	$(ESC)[0m

# Colors
BLACK		:=	$(ESC)[0;30m
RED			:=	$(ESC)[0;31m
GREEN		:=	$(ESC)[0;32m
YELLOW		:=	$(ESC)[0;33m
BLUE		:=	$(ESC)[0;34m
MAGENTA		:=	$(ESC)[0;35m
CYAN		:=	$(ESC)[0;36m
WHITE		:=	$(ESC)[0;37m
DEFAULT		:=	$(ESC)[0;39m

# Text Styles
BOLD		:=	$(ESC)[1m
DIM			:=	$(ESC)[2m
ITALIC		:=	$(ESC)[3m
UNDERLINE	:=	$(ESC)[4m
INVERSE		:=	$(ESC)[7m
HIDDEN		:=	$(ESC)[8m
STRIKE		:=	$(ESC)[9m

# =========================================
# Compilation Targets
# =========================================

PYTHON	=	python3
SRC		=	src


install:
	@printf "${BOLD}${BLUE}Synchronizing packages${RESET}\n"
	@uv sync

run:
	@uv run ${PYTHON} -O -m ${SRC}

limit:
	@systemd-run --user --scope \
  	--property MemoryMax=4G \
  	nice --adjustment=19 uv run ${PYTHON} -m ${SRC}

moulinette:
	export UV_PROJECT_ENVIRONMENT = ($$UV_PROJECT_ENVIRONMENT)_moulinette

debug:
	@printf "${INVERSE}${YELLOW}Running in DEBUG mode${RESET}\n"
	@uv run ${PYTHON} -m pdb -m ${SRC}

clean:
	@printf "${YELLOW}Cleaning artifacts${RESET}\n"
	@find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +

lint:
	@printf "${GREEN}${BOLD}Linting${RESET}\n"
	@printf "${GREEN}Syntax ${CYAN}(flake8)${RESET}\n"
	@uv run flake8 ${SRC} || true
	@printf "${GREEN}Typing ${CYAN}(mypy)${RESET}\n"
	@uv run mypy ${SRC} \
	--warn-return-any \
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs \
	|| true

lint-strict:
	@printf "${GREEN}${BOLD}Linting${RESET}\n"
	@printf "${GREEN}Syntax ${CYAN}(flake8)${RESET}\n"
	@uv run python3 -m flake8 ${SRC} || true
	@printf "${GREEN}Typing ${CYAN}(mypy ${RED}--strict${CYAN})${RESET}\n"
	@uv run python3 -m mypy ${SRC} --strict || true

.PHONY: install run debug clean lint lint-strict
