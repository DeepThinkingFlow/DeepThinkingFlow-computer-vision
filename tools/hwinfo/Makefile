# dtflowcv Hardware Tools — Makefile
#
# Targets:
#   make all         — Build C tools + Java profiler
#   make hwguide     — Build standalone C hardware guide
#   make java        — Compile Java profiler
#   make check       — Run hardware check (shell script)
#   make clean       — Remove build artifacts
#   make install     — Copy binaries to project bin/
#
# Works on Linux and macOS with GCC/Clang installed.
# Java compilation optional (only if javac is available).
#

CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra -Wpedantic -std=c11
JAVAC    ?= javac
BINDIR   ?= ../../bin

.PHONY: all hwguide java check clean install

all: hwguide java

# ── C standalone hardware guide ──────────────────────────

hwguide: hwguide_standalone.c
	$(CC) $(CFLAGS) -o dtflowcv_hwguide hwguide_standalone.c
	@echo "[OK] Built dtflowcv_hwguide"

# ── Java cross-platform profiler ─────────────────────────

java: HardwareProfiler.java
	@if command -v $(JAVAC) >/dev/null 2>&1; then \
		$(JAVAC) HardwareProfiler.java && echo "[OK] Compiled HardwareProfiler.class"; \
	else \
		echo "[SKIP] javac not found — Java profiler not compiled"; \
	fi

# ── Quick hardware check ────────────────────────────────

check:
	@bash hwcheck.sh

# ── Install ─────────────────────────────────────────────

install: all
	@mkdir -p $(BINDIR)
	@cp -v dtflowcv_hwguide $(BINDIR)/ 2>/dev/null || true
	@cp -v HardwareProfiler.class $(BINDIR)/ 2>/dev/null || true
	@cp -v hwcheck.sh $(BINDIR)/ 2>/dev/null || true
	@echo "[OK] Installed to $(BINDIR)/"

# ── Clean ───────────────────────────────────────────────

clean:
	rm -f dtflowcv_hwguide HardwareProfiler.class
	@echo "[OK] Cleaned"
