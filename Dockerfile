FROM python:3.10-slim
#using a small base image

#setting working directory inside container
WORKDIR /app 

RUN apt-get update && apt-get install -y \
    libgl1 \ 
    libglib2.0-0 \ 
&& rm -rf /var/lib/apt/lists/*

#copy requirements and install dependencies 
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#copying the rest of the project files 
COPY . . 

#EXPOSING FLASK PORT 
EXPOSE 5000

CMD ["python","app.py"]



