FROM python:3.10.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENV PORT=1234
EXPOSE 1234
CMD ["python", "api.py"]
