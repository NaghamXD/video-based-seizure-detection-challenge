# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# preven python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# keep python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# create a non-privileged user that the app will run under
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# download dependencies in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# switch to the non-privileged user to run the application
USER appuser

# create data volumes
VOLUME ["/data"]
VOLUME ["/output"]

# define environment variables
ENV INPUT=""
ENV OUTPUT=""

# run the application
CMD python3 -m main "/data/$INPUT" "/output/$OUTPUT"
