"""rubin – App Launcher.

Startet den Flask-Backend-Server, der die React-UI served
und alle API-Endpoints bereitstellt.
"""
from app.server import main

if __name__ == "__main__":
    main()
