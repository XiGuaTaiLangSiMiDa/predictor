#!/bin/bash

# Activate virtual environment
source venv/bin/activate

SUPERVISOR_PID_FILE="/Users/yafan/workspace/tql/supervisord.pid"
SOCK_FILE="/tmp/supervisor.sock"

cleanup() {
    # Kill any existing supervisord process
    if [ -f "$SUPERVISOR_PID_FILE" ]; then
        pid=$(cat "$SUPERVISOR_PID_FILE")
        if ps -p $pid > /dev/null; then
            kill $pid
        fi
        rm -f "$SUPERVISOR_PID_FILE"
    fi
    
    # Remove socket file if it exists
    if [ -S "$SOCK_FILE" ]; then
        rm -f "$SOCK_FILE"
    fi
    
    # Kill any remaining gunicorn processes
    pkill -f "gunicorn"
}

check_process() {
    if [ -f "$SUPERVISOR_PID_FILE" ] && ps -p $(cat "$SUPERVISOR_PID_FILE") > /dev/null; then
        return 0
    else
        return 1
    fi
}

start_supervisor() {
    supervisord -c supervisor.conf
    sleep 2  # Give supervisord time to start
}

case "$1" in
    "start")
        echo "Starting crypto predictor server..."
        if check_process; then
            echo "Server is already running. Use 'restart' to restart it or 'status' to check its status."
        else
            cleanup
            start_supervisor
            supervisorctl -c supervisor.conf start crypto_predictor
            echo "Server started successfully."
        fi
        ;;
    "stop")
        echo "Stopping crypto predictor server..."
        if check_process; then
            supervisorctl -c supervisor.conf stop crypto_predictor
            supervisorctl -c supervisor.conf shutdown
            cleanup
            echo "Server stopped successfully."
        else
            cleanup
            echo "Server was not running, cleaned up any remaining processes."
        fi
        ;;
    "restart")
        echo "Restarting crypto predictor server..."
        $0 stop
        sleep 2
        $0 start
        ;;
    "status")
        if ! check_process; then
            echo "Server is not running."
        else
            echo "Server status:"
            supervisorctl -c supervisor.conf status
        fi
        ;;
    "logs")
        echo "Showing crypto predictor server logs..."
        if [ -f "crypto_predictor.out.log" ]; then
            tail -f crypto_predictor.out.log
        else
            echo "Log file not found. Make sure the server has been started at least once."
        fi
        ;;
    "install")
        echo "Installing dependencies..."
        pip install -r requirements.txt
        pip install supervisor
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|install}"
        exit 1
        ;;
esac

exit 0
