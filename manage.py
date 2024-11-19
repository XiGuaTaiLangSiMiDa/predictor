#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import time

class ServerManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.supervisor_conf = os.path.join(self.base_dir, 'supervisor.conf')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.pid_file = os.path.join(self.logs_dir, 'supervisord.pid')
        self.sock_file = '/tmp/supervisor.sock'
        self.venv_dir = os.path.join(self.base_dir, 'venv')
        self.venv_bin = os.path.join(self.venv_dir, 'bin')

    def ensure_dirs(self):
        """Ensure required directories exist"""
        os.makedirs(self.logs_dir, exist_ok=True)

    def get_command_path(self, cmd):
        """Get full path to command in virtualenv"""
        return os.path.join(self.venv_bin, cmd)

    def cleanup(self):
        """Clean up processes and files"""
        print("Performing cleanup...")
        
        # Kill supervisord process if running
        if os.path.exists(self.pid_file):
            try:
                with open(self.pid_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                pass
            try:
                os.remove(self.pid_file)
            except FileNotFoundError:
                pass

        # Remove socket file
        try:
            if os.path.exists(self.sock_file):
                os.remove(self.sock_file)
        except FileNotFoundError:
            pass

        # Kill any remaining processes
        subprocess.run("pkill -f supervisord", shell=True)
        subprocess.run("pkill -f gunicorn", shell=True)
        
        # Wait for processes to fully terminate
        time.sleep(2)

    def start(self):
        """Start the server"""
        print("Starting server...")
        self.ensure_dirs()
        self.cleanup()
        
        supervisord = self.get_command_path('supervisord')
        supervisorctl = self.get_command_path('supervisorctl')
        
        # Start supervisord
        subprocess.run(f"{supervisord} -c {self.supervisor_conf}", shell=True)
        time.sleep(2)  # Give supervisord time to start
        
        # Start the application
        subprocess.run(f"{supervisorctl} -c {self.supervisor_conf} start crypto_predictor", shell=True)
        print("Server started successfully")

    def stop(self):
        """Stop the server"""
        print("Stopping server...")
        supervisorctl = self.get_command_path('supervisorctl')
        
        try:
            subprocess.run(f"{supervisorctl} -c {self.supervisor_conf} stop crypto_predictor", shell=True)
            subprocess.run(f"{supervisorctl} -c {self.supervisor_conf} shutdown", shell=True)
        except Exception as e:
            print(f"Warning: {e}")
        
        self.cleanup()
        print("Server stopped successfully")

    def restart(self):
        """Restart the server"""
        print("Restarting server...")
        self.stop()
        time.sleep(2)
        self.start()

    def status(self):
        """Check server status"""
        supervisorctl = self.get_command_path('supervisorctl')
        try:
            subprocess.run(f"{supervisorctl} -c {self.supervisor_conf} status", shell=True)
        except Exception as e:
            print(f"Error checking status: {e}")
            print("Server might not be running")

    def logs(self):
        """Show server logs"""
        log_file = os.path.join(self.logs_dir, 'crypto_predictor.out.log')
        if os.path.exists(log_file):
            subprocess.run(f"tail -f {log_file}", shell=True)
        else:
            print("Log file not found. Make sure the server has been started.")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['start', 'stop', 'restart', 'status', 'logs']:
        print("Usage: python manage.py {start|stop|restart|status|logs}")
        sys.exit(1)

    manager = ServerManager()
    command = sys.argv[1]

    if command == 'start':
        manager.start()
    elif command == 'stop':
        manager.stop()
    elif command == 'restart':
        manager.restart()
    elif command == 'status':
        manager.status()
    elif command == 'logs':
        manager.logs()

if __name__ == '__main__':
    main()
