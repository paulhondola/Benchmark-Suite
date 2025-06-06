# Start from official slim Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install OS-level dependencies needed for numpy, pandas, psycopg2, etc.
RUN apt-get update && apt-get install -y \
	gcc \
	clang \
	libpq-dev \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Copy all project files into the container
COPY . .

# Upgrade pip + install your benchmark suite with platform extras
RUN pip install --upgrade pip
RUN pip install .
