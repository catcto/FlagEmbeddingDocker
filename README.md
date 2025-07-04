# FlagEmbeddingDocker

This repository provides a Docker image for [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), enabling you to deploy the FlagEmbedding server within a Docker container.

## Usage

To run this Docker container, you’ll need a machine with NVIDIA GPU support and the NVIDIA Container Toolkit installed. For detailed installation steps, please refer to the [NVIDIA Container Toolkit](https://notes.xiaowu.ai/%E5%BC%80%E5%8F%91%E7%AC%94%E8%AE%B0/AI/NVIDIA#%E5%AE%89%E8%A3%85+NVIDIA+Container+Toolkit) guide.

### Build the Docker image

```shell
$ docker build -t flagembedding .
```

### Using docker command

```shell
$ docker run -d --name flagembedding_server -p 8080:8080 \
         --runtime=nvidia \
         -e NVIDIA_DRIVER_CAPABILITIES=all \
         -e NVIDIA_VISIBLE_DEVICES=all \
         flagembedding
```

### Using docker compose

1. Create a `docker-compose.yml` file:
```yaml
services:
  flagembedding_server:
    image: flagembedding
    container_name: flagembedding_server
    ports:
      - "8080:8080"
    restart: always
    runtime: nvidia
    environment:
      NVIDIA_DRIVER_CAPABILITIES: all
      NVIDIA_VISIBLE_DEVICES: all
```
2. Start the container:
```shell
$ docker compose up -d
```

## Testing

To test the API, use `curl`:

```shell
curl -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-large-zh-v1.5",
    "texts": ["今天天气很好", "你好，世界"]
  }'
```