"""Run the web UI with Flask's own server, debug OFF (Task 4.4).

    python run.py                      # http://127.0.0.1:5000

Debug mode is off because the Werkzeug debugger lets anyone who can see an error page
execute Python on the server, it leaks source code and settings in tracebacks, and the
auto-reloader loads the model twice. In Docker we use gunicorn, a production WSGI
server, instead of this development server.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=False)
