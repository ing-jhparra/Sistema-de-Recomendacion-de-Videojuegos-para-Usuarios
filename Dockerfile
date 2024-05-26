FROM python:3.9

WORKDIR /code

RUN pip install fastapi[all] uvicorn[standard]

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

RUN rm -rf /code/img 

RUN rm -rf /code/notebooks 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]