UV := uv run --no-sync

.DEFAULT_GOAL := help
.PHONY: help setup format lint typecheck test test_rf5 build clean convert-weights

help:
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "                         FOMO Makefile"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Development Commands:"
	@echo "  setup                         - Create venv and install package + dev dependencies"
	@echo "  format                        - Format code with ruff"
	@echo "  lint                          - Run linter"
	@echo "  typecheck                     - Run type checker"
	@echo "  test                          - Build package and run tests on Modal (GPU)"
	@echo "  test_rf5                      - Run RF5 training benchmark tests"
	@echo "  build                         - Build package"
	@echo "  convert-weights                - Convert Keras MobileNetV2 weights → FOMOs/m/l.pt via Modal"
	@echo "  clean                         - Remove build and test cache artifacts"

setup:
	uv sync --dev
	@echo ""
	@echo "✅ Setup complete! To activate the virtual environment, run:"
	@echo "   source .venv/bin/activate"

format:
	$(UV) ruff format

lint:
	$(UV) ruff check --fix

typecheck:
	$(UV) ty check

test:
	rm -rf dist/
	uv build --out-dir dist/
	modal run tests/modal_tests.py

convert-weights:
	modal run scripts/modal_convert_weights.py

test_rf5: clean
	$(UV) pytest tests/e2e/test_rf5_training.py -m rf5 -v

build:
	@echo "📦 Building package..."
	@mkdir -p dist
	uv build --out-dir dist/
	@echo "✅ Package built:"
	@ls -lh dist/*.whl

clean:
	@echo "🧹 Cleaning build and test cache artifacts..."
	@rm -rf dist *.egg-info .ruff_cache .pytest_cache
	@rm -rf /tmp/pytest-of-$(USER) 2>/dev/null || true
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "✅ Clean complete!"
