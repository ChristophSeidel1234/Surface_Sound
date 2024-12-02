FROM python:3.10-slim

WORKDIR /src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "streamlit","run","app.py" , "--server.address","0.0.0.0"]
