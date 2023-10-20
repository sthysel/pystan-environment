FROM python:3.11

RUN apt-get update && apt-get install -y \
    libpython3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pystan numpy
COPY esky-estimator.py .
ENTRYPOINT ["python", "esky-estimator.py"]
