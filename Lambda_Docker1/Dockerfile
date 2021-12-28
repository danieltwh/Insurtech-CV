FROM public.ecr.aws/lambda/python:3.7.2021.12.22.00
ADD . ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
# WORKDIR /insurtech-cv
# RUN yum update -y
# RUN yum -y install ffmpeg libsm6 libxext6
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
# ENTRYPOINT [ "python" ]
CMD [ "app2.predict" ]