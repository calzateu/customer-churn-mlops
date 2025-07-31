# .env can define ports like MLFLOW_PORT=5050

up:
	docker compose up --build

down:
	docker compose down

restart:
	docker compose down && docker compose up --build

logs:
	docker compose logs -f

# Complete cleanup: volumes, images, network
clean:
	docker compose down --volumes --remove-orphans
	docker system prune -f

# See services status
ps:
	docker compose ps

