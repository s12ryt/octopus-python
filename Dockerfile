FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY octopus_python ./octopus_python
COPY static ./static
RUN pip install --no-cache-dir .

EXPOSE 8080
VOLUME ["/app/data"]
CMD ["octopus-python", "start"]
