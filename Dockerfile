FROM python:3.12

USER root

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install fastapi && pip install uvicorn

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/

RUN rm -rf /code/img 

RUN rm -rf /code/notebooks 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]