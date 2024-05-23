# syntax=docker/dockerfile:1
FROM python:3.10

# install app dependencies
WORKDIR /
COPY requirements.txt /
RUN target=/root/.cache/pip \
    pip3 install -r requirements.txt

# Copy the config.ini file into the working directory
COPY config.ini /

# install app
COPY server.py /

# final configuration
# ENV FLASK_APP=hello
EXPOSE 5001
CMD ["python", "server.py"]