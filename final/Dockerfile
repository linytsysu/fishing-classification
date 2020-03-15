FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

WORKDIR /
ADD Dockerfile run.sh main.py parameters.py main_v3.py requirements.txt /
RUN pip --no-cache-dir install  -r requirements.txt

CMD ["sh", "run.sh"]
