# Multi-stage Dockerfile for AI Governance Dashboard
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jdk \
    wget \
    curl \
    ca-certificates \
    python3-tk \
    tk-dev \
    && rm -rf /var/lib/apt/lists/*

# Find Java installation and set JAVA_HOME
RUN if [ -d "/usr/lib/jvm/default-java" ]; then \
        export JAVA_HOME=/usr/lib/jvm/default-java; \
    elif [ -d "/usr/lib/jvm/java-11-openjdk-amd64" ]; then \
        export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64; \
    else \
        JAVA_PATH=$(readlink -f $(which java)); \
        export JAVA_HOME=$(dirname $(dirname $JAVA_PATH)); \
    fi && \
    echo "JAVA_HOME=$JAVA_HOME" >> /etc/environment

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

# Verify Java installation
RUN java -version || (echo "Java verification failed" && exit 1)

# Install Apache Spark
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget --no-check-certificate -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    chmod +x /opt/spark/bin/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/results /app/data

# Expose ports
EXPOSE 8501 8080 4040

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
