#!/bin/bash
# MLflow Server Management Script

set -e
LOCALHOST=127.0.0.1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

case "$1" in
    start)
        echo "🚀 Starting MLflow server..."
        uv run python scripts/start_mlflow.py start "${@:2}"
        ;;
    status)
        echo "🔍 Checking MLflow server status..."
        uv run python scripts/start_mlflow.py status "${@:2}"
        ;;
    stop)
        echo "🛑 Stopping MLflow server..."
        MLFLOW_PORT=$(grep -o 'mlflow_tracking_uri.*localhost:[0-9]*' src/config/settings.py | grep -o '[0-9]*' || echo "5000")
        ps -A | grep gunicorn | grep ${LOCALHOST}:${MLFLOW_PORT} | cut -d' ' -f1 | xargs kill
        if `ps -A | grep gunicorn | grep ${LOCALHOST}:${MLFLOW_PORT}` ; then
            echo "Failed to stop MLflow server"
            exit 1
        fi
        echo "🛑 MLflow server stopped successfully 🛑"
        ;;
    restart)
        echo "🔄 Restarting MLflow server..."
        "$0" stop
        sleep 2
        "$0" start "${@:2}"
        ;;
    ui)
        MLFLOW_PORT=$(grep -o 'mlflow_tracking_uri.*localhost:[0-9]*' src/config/settings.py | grep -o '[0-9]*' || echo "5000")
        echo "🌐 Opening MLflow UI at http://localhost:${MLFLOW_PORT}"
        open "http://localhost:${MLFLOW_PORT}" 2>/dev/null || echo "Open http://localhost:${MLFLOW_PORT} in your browser"
        ;;
    *)
        echo "MLflow Server Management"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|ui} [options]"
        echo ""
        echo "Commands:"
        echo "  start    Start MLflow server"
        echo "  stop     Stop MLflow server"
        echo "  restart  Restart MLflow server" 
        echo "  status   Check server status"
        echo "  ui       Open MLflow web interface"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start with default settings"
        echo "  $0 start --port 5001        # Start on custom port"
        echo "  $0 start --dev              # Start in development mode"
        echo "  $0 status                   # Check if server is running"
        exit 1
        ;;
esac