# Base image: slim keeps the image small while still having a full Python runtime.
FROM python:3.11-slim

# Set a working directory inside the container so paths are predictable.
WORKDIR /app

# Copy only the dependency list first so Docker can cache this layer and skip
# re-installing packages when only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the versioned retrieval index. docs/ and
# ingest.py aren't needed at runtime — ingestion already happened at commit
# time, the same way Week 1's ML lab doesn't ship prototype.py/train_model.py.
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# Document the port the service listens on (informational; doesn't publish it).
EXPOSE 8000

# Run the service with uvicorn. --host 0.0.0.0 is required so the server is
# reachable from outside the container, not just from localhost inside it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
