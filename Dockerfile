FROM ubuntu:18.04

ENV TZ=America/Sao_Paulo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
&& apt-get install -y python3-dev python3-opencv python3-pip

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requeriments.txt

EXPOSE 5000

ENTRYPOINT ["python3"]
CMD ["run.py"]
