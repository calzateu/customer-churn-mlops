# .env can define ports like MLFLOW_PORT=5050

up:
	mkdir -p ./mlartifacts
	docker compose up --build

down:
	docker compose down

down-volumes:
	docker compose down --volumes
	rm -rf ./mlartifacts

restart:
	docker compose down && docker compose up --build

logs:
	docker compose logs -f

# Complete cleanup: volumes, images, network, folders
clean:
	docker compose down --volumes --remove-orphans
	docker system prune -f
	rm -rf ./mlartifacts

# See services status
ps:
	docker compose ps

