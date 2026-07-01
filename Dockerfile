FROM python:3.12

WORKDIR /preliminary_analysis_tool

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY . .

CMD ["streamlit", "run", "main.py"]