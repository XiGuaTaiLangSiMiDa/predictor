#!/bin/bash

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "venv" ]; then
        echo "Virtual environment not found. Creating one..."
        python3 -m venv venv
    fi
}

# Function to activate virtual environment
activate_venv() {
    source venv/bin/activate
}

# Function to install dependencies
install_deps() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Function to create necessary directories
create_dirs() {
    mkdir -p logs
    mkdir -p app/static/plots
    chmod 777 app/static/plots
}

# Function to check server status
check_status() {
    if [ -f logs/server.pid ]; then
        PID=$(cat logs/server.pid)
        if ps -p $PID > /dev/null; then
            echo "Server is running (PID: $PID)"
            echo "Process info:"
            ps -p $PID -o pid,ppid,user,%cpu,%mem,start,time,command
            echo -e "\nListening ports:"
            lsof -i :5000 | grep LISTEN
            echo -e "\nLast 5 log entries:"
            tail -n 5 logs/server.log
        else
            echo "Server not running (stale PID file found)"
            rm logs/server.pid
        fi
    else
        echo "Server not running"
    fi
}

case "$1" in
    "start")
        if [ -f logs/server.pid ]; then
            echo "Server already running (PID: $(cat logs/server.pid))"
            exit 1
        fi
        check_venv
        activate_venv
        create_dirs
        echo "Starting server..."
        FLASK_DEBUG=1 python3 -m flask run --host=0.0.0.0 --port=5000 >> logs/server.log 2>&1 &
        echo $! > logs/server.pid
        echo "Server started. PID: $(cat logs/server.pid)"
        sleep 2
        check_status
        ;;
        
    "stop")
        if [ -f logs/server.pid ]; then
            echo "Stopping server..."
            kill $(cat logs/server.pid)
            rm logs/server.pid
            echo "Server stopped"
        else
            echo "Server not running"
        fi
        ;;
        
    "restart")
        $0 stop
        sleep 2
        $0 start
        ;;

    "status")
        check_status
        ;;
        
    "log")
        if [ "$2" = "error" ]; then
            echo "Showing error logs (Ctrl+C to exit):"
            tail -f logs/app.log
        else
            echo "Showing server logs (Ctrl+C to exit):"
            tail -f logs/server.log
        fi
        ;;
        
    "install")
        check_venv
        activate_venv
        install_deps
        create_dirs
        echo "Installation complete"
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status|log|install}"
        echo "  start   : Start the server"
        echo "  stop    : Stop the server"
        echo "  restart : Restart the server"
        echo "  status  : Check server status"
        echo "  log     : View server logs (use 'log error' for error logs)"
        echo "  install : Install dependencies and set up environment"
        exit 1
        ;;
esac

exit 0
