FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

EXPOSE 8080

# Azure Container Apps routes to a fixed --target-port instead of injecting PORT,
# so default it here. The ${PORT:-8080} form also works on App Service and ACI,
# which do inject PORT.
CMD ["sh", "-c", "uvicorn main_azure:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
