FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt merge_stl.py ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "merge_stl.py"]
