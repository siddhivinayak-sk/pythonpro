# =============================================================================
# kiro-security-scan
# Packages the single-file Python scanner + the Kiro headless CLI.
# Run with just KIRO_API_KEY set in the environment.
#
# Build:
#   docker build -t kiro-security-scan .
#
# Run (scan a mounted directory):
#   docker run --rm \
#     -e KIRO_API_KEY="$KIRO_API_KEY" \
#     -v /path/to/artifacts:/scan/input:ro \
#     -v /path/to/output:/scan/out \
#     kiro-security-scan \
#     --input /scan/input \
#     --report /scan/out/report.html \
#     --yaml-out /scan/out/findings.yaml
#
# Exit codes: 0 = PASS, 1 = FAIL (open critical/high), 2 = setup error.
# =============================================================================

# glibc 2.34+ is required by the Kiro CLI; Debian bookworm ships 2.36.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KIRO_LOG_NO_COLOR=1 \
    PATH="/root/.local/bin:${PATH}"

# --- System deps + Kiro CLI -------------------------------------------------
# curl/ca-certificates for the installer; git often needed by the CLI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl ca-certificates git unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://cli.kiro.dev/install | bash \
    && ( command -v kiro-cli >/dev/null 2>&1 && kiro-cli --version || true )

# --- Python deps ------------------------------------------------------------
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- App --------------------------------------------------------------------
COPY main.py ./
COPY config.yaml whitelist.yaml ./

# Default working area for mounted inputs/outputs.
RUN mkdir -p /scan/input /scan/out
VOLUME ["/scan/input", "/scan/out"]

# KIRO_API_KEY must be provided at runtime (-e KIRO_API_KEY=...).
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--input", "/scan/input", \
     "--report", "/scan/out/kiro-security-report.html", \
     "--yaml-out", "/scan/out/kiro-security-findings.yaml"]
