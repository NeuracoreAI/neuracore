FROM python:3.10-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install ".[dev,ml]"

CMD ["pytest", "tests/unit", "--cov=neuracore"]
