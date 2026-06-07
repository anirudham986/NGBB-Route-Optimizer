FROM python:3.11-slim

# Install SCIP dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgmp-dev \
    libreadline-dev \
    zlib1g-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install SCIP 8.x
RUN wget -q https://scipopt.org/download/release/SCIPOptSuite-8.0.3-Linux-ubuntu.deb && \
    dpkg -i SCIPOptSuite-8.0.3-Linux-ubuntu.deb || apt-get install -f -y && \
    rm SCIPOptSuite-8.0.3-Linux-ubuntu.deb

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["bash"]
