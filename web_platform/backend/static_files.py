"""
Static file serving for the frontend
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

def setup_static_files(app: FastAPI):
    """Setup static file serving for frontend"""
    
    # Get the frontend directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(os.path.dirname(current_dir), 'frontend')
    
    # Serve frontend files
    if os.path.exists(frontend_dir):
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        
        # Serve SPA dashboard at /frontend
        @app.get("/frontend")
        @app.get("/frontend/")
        async def serve_frontend():
            spa_path = os.path.join(frontend_dir, 'dashboard_spa.html')
            if os.path.exists(spa_path):
                return FileResponse(spa_path)
            # Fallback to old index if SPA doesn't exist
            index_path = os.path.join(frontend_dir, 'index.html')
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"error": "Frontend not found"}
        
        # Serve dashboard.html (for backward compatibility)
        @app.get("/frontend/dashboard.html")
        async def serve_dashboard():
            spa_path = os.path.join(frontend_dir, 'dashboard_spa.html')
            if os.path.exists(spa_path):
                return FileResponse(spa_path)
            dashboard_path = os.path.join(frontend_dir, 'dashboard.html')
            if os.path.exists(dashboard_path):
                return FileResponse(dashboard_path)
            return {"error": "Dashboard not found"}
        
        # Redirect patterns.html to SPA with hash
        @app.get("/frontend/patterns.html")
        async def serve_patterns():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/frontend/#patterns", status_code=302)
        
        # Redirect performance.html to SPA with hash
        @app.get("/frontend/performance.html")
        async def serve_performance():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/frontend/#performance", status_code=302)
    
    return app