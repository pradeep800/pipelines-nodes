FROM python:3.11-slim

WORKDIR /app

# Install app dependencies
RUN pip install --no-cache-dir streamlit vedo boto3

COPY viz2d3d.py .

EXPOSE 8501

CMD ["streamlit", "run", "viz2d3d.py", "--server.address=0.0.0.0", "--server.port=8501"]
