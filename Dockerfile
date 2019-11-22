FROM python:3.7 as build_image

WORKDIR /credit_collections_rl
COPY . /credit_collections_rl/

RUN pip install --upgrade pip && \
    pip install pipenv && \
    pipenv install --system && \
    pip list

CMD cd credit_collections_rl
