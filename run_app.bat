@echo off
cd /d "%~dp0"
call appenv\Scripts\activate

:: Start the Flask apps
start cmd /k python finalapp.py


:: Wait for a few seconds to ensure the server starts (adjust as needed)
timeout /t 5 /nobreak

:: Open the default web browser with your Flask app URL (adjust port if necessary)
start "" http://127.0.0.1:5001

exit
