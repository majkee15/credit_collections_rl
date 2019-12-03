FROM python:3.7 as build_image

WORKDIR /credit_collections_rl

COPY requirements.txt /credit_collections_rl/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip list

COPY . /credit_collections_rl/

CMD cd credit_collections_rl

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python","bruteforce_search.py"]
