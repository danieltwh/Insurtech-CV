FROM python:3.7.11
ADD . /insurtech-cv
WORKDIR /insurtech-cv
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

# ENTRYPOINT [ "python" ]
# CMD [ "app.py" ]

ENTRYPOINT ["./gunicorn_starter.sh"]