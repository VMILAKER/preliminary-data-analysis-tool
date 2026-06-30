FROM python:3.12

WORKDIR /PAT

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY ./ /PAT/

ENTRYPOINT ["streamlit", "run", "main.py"]