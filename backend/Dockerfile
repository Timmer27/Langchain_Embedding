# syntax=docker/dockerfile:1
# FROM python:3.9-slim-bullseye
# FROM python:3.10
FROM python:3.10-slim-bullseye
WORKDIR /

# Install the latest sqlite3
RUN apt-get update && \
    apt-get install -y wget build-essential libsqlite3-dev && \
    wget https://www.sqlite.org/2024/sqlite-autoconf-3450300.tar.gz && \
    tar xzf sqlite-autoconf-3450300.tar.gz && \
    cd sqlite-autoconf-3450300 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3450300 sqlite-autoconf-3450300.tar.gz

# Update the library path
ENV LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

# install app dependencies
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

# RUN apt-get update && apt-get install -y python3 python3-pip

# Copy the config.ini file into the working directory
COPY config.ini /
COPY src /src
COPY data /data

# Set the PYTHONPATH
ENV PYTHONPATH="/src"

# install app
COPY server.py /

# final configuration
# ENV FLASK_APP=hello
EXPOSE 5001
CMD ["python", "server.py"]