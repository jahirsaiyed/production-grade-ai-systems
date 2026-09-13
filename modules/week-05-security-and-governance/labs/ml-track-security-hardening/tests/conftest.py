import os

os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-for-production"
os.environ["MODEL_ENCRYPTION_KEY"] = "kR9mZ3xQhT7vN2pL8wF5yB1cA6dE4gJ0sU3iO9nM7k8="
os.environ["FEATURE_STORE_FAILURE_RATE"] = "0"
