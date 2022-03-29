FROM python:3.10.2
COPY . .
RUN pip install -r requirements.txt
CMD [ "python" ,"wsgi.py" ]
