FROM python:3.12-slim

WORKDIR /preliminary_analysis_tool

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MPLBACKEND=Agg

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt


COPY . .

EXPOSE 8502

HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['STREAMLIT_SERVER_PORT'] + '/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "main.py"]
