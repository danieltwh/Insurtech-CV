FROM python:3.7.11
ADD . /insurtech-cv
WORKDIR /insurtech-cv
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements2.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]