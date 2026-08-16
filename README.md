# Streamlit preliminary data analysis tool
This version of application supports analytics and ML engineers conduct preliminary data analysis by Pandas or PyGWalker. 

As export files user can download cleaned .csv file without duplicates or review in .pdf format.

! *Application might be deployed locally or with Docker container.*

## Fast start
- With Docker
```
# 1. Copy .env.example to .env
cp .env.example .env

# 2. Build Docker container
docker compose build

# 3. Pick up the container
docker compose up
```
- By requirements.txt
```
# 1. Copy .env.example to .env
cp .env.example .env

# 2. Install requirements.txt via pip
pip install requirements.txt

# 3. Run streamlit
streamlit run main.py --server.port 8502
```
