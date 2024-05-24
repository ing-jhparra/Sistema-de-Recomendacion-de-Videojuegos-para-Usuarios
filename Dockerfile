FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/

RUN rm /code/img -rf

RUN rm /code/Datasets -rf

RUN rm /code/notebooks -rf

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]