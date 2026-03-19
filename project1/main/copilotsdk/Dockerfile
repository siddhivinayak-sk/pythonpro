FROM python:3.13-slim-bookworm
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && curl -fsSL https://deb.nodesource.com/setup_23.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g --omit=dev --no-audit --no-fund @github/copilot \
    && npm cache clean --force \
    && rm -rf /root/.npm /tmp/npm-*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
RUN node --version && npm --version && python --version && copilot --version
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 -m -d /home/appuser appuser
WORKDIR /app
COPY my_copilot.py .
RUN chown appuser:appgroup my_copilot.py && chmod 550 my_copilot.py
USER appuser
ENTRYPOINT ["python", "my_copilot.py"]

# docker build -t mr-review project1/main/copilotsdk/
# docker run --rm -e GITLAB_TOKEN=$env:GITLAB_API_TOKEN -e GITLAB_HOST=$env:MY_GITLAB_LINK -e COPILOT_GITHUB_TOKEN=$env:_COPILOT_GITHUB_TOKEN -e MR_URL=""  mr-review
