# Setup

To run the examples of the libraries we put here, we strongly recommend using Docker or a specific Virtualenv.

First, let's clone the repository : 

````bash
git clone https://github.com/gfalcone/mlserve
cd mlserve
````

Let's get all the dependencies : 

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

Now let's create our development environment : 

````bash
bash create_dev_environment.sh
````

Now we can train our models and serve them !