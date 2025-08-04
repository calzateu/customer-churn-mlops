
.DEFAULT_GOAL := help

# Variables (optional: load from .env if needed)
MLARTIFACTS_DIR := ./mlartifacts

# ----------------------
# Commands
# ----------------------

## up            : Build and start all services (creates mlartifacts folder)
up: $(MLARTIFACTS_DIR)  ## ⬆️  Start docker-compose stack
	docker compose up --build

## down          : Stop all services
down:           ## ⬇️  Stop docker-compose stack
	docker compose down

## down-volumes  : Stop all services and remove volumes
down-volumes:   ## ⚠️  Stop & remove volumes (also deletes artifacts)
	docker compose down --volumes
	rm -rf $(MLARTIFACTS_DIR)

## restart       : Restart all services (clean build)
restart:        ## ♻️  Restart docker-compose stack
	docker compose down && docker compose up --build

## logs          : Follow logs from all services
logs:           ## 📜 Tail logs from services
	docker compose logs -f

## clean         : Full cleanup (volumes, images, orphans, artifacts)
clean:          ## 🧹 Full cleanup (containers + volumes + artifacts)
	docker compose down --volumes --remove-orphans
	docker system prune -f
	rm -rf $(MLARTIFACTS_DIR)

## ps            : Show status of all containers
ps:             ## 📦 Show running containers
	docker compose ps

## help          : Show this help message
help:           ## 🆘 Show help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ----------------------
# Helpers
# ----------------------

$(MLARTIFACTS_DIR):
	@mkdir -p $(MLARTIFACTS_DIR)
