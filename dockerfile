FROM pathwaycom/pathway:latest

WORKDIR /app
COPY . /app

RUN pip install google-generativeai python-dotenv

CMD ["python", "app.py"]
