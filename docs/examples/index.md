# Setup

To run the examples of the libraries we put here, we strongly recommend using Docker

First, let's clone the repository, build the docker container and enter it: 

````bash
git clone https://github.com/gfalcone/mlserve
cd mlserve
docker build --tag=mlserve .
docker run -ti -p 8000:8000 mlserve bash
````

Now we can train our models and serve them !