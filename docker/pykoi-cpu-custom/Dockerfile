FROM pykoi/pykoi-cpu:base

# set the working directory in the container
WORKDIR /app

# copy the content
COPY . /app

# Expose the port on which the app will be running 
EXPOSE 5000

# For dev
CMD ["python", "app.py"]
