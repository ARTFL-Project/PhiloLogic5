"""Rebuild a database's web app from the installed one, as after an upgrade. (A database copied to another server
needs no rebuild: its web app is built for no host or URL prefix in particular.)"""
import sys
import os


if __name__ == "__main__":
    philo_db = sys.argv[1]
    app_path = f"{philo_db}/app"
    os.system(f"rm -rf {app_path}")
    os.system(f"cp -R /var/lib/philologic5/web_app/app {philo_db}/")
    os.system(f"chown -R $(whoami) {app_path}")  # Make sure we have the correct permissions for npm to run
    npm = "/var/lib/philologic5/bin/npm"  # as the loader runs it: the copy has no node_modules
    os.system(f"cd {app_path}; {npm} install && {npm} run build")
    print(f"{philo_db} done")
