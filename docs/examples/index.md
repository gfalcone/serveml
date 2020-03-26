# Setup

To run the examples of the libraries we put here, we strongly recommend using Docker

First, let's clone the repository, and setup MLflow container 

````bash
git clone https://github.com/gfalcone/serveml
cd serveml
mkdir -p /tmp/mlflow
docker-compose build
docker-compose up
````

Now we can train our models and serve them !

When you're done testing, don't forget to kill the MLflow container with : 

```bash
docker-compose down
```