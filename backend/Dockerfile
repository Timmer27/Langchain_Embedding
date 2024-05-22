# syntax=docker/dockerfile:1
FROM python:3.9-slim-bullseye

# install app dependencies
WORKDIR /
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# RUN apt-get update && apt-get install -y python3 python3-pip

# Copy the config.ini file into the working directory
COPY config.ini /

# install app
COPY server.py /

# final configuration
# ENV FLASK_APP=hello
EXPOSE 5001
CMD ["python", "server.py"]