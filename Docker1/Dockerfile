FROM python:3.7.11
ADD . /insurtech-cv
WORKDIR /insurtech-cv
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]