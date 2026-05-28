#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DocMind RAG Q&A — One-shot startup script
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       🧠 DocMind · RAG Document Q&A       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌  Error: ANTHROPIC_API_KEY environment variable is not set."
    echo "    Export it first:  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi
echo "✅  ANTHROPIC_API_KEY found"

# Install backend deps
echo ""
echo "📦  Installing backend dependencies..."
cd "$PROJECT_ROOT/backend"
pip install -r requirements.txt -q

# Install frontend deps
echo "📦  Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
pip install -r requirements.txt -q

echo ""
echo "🚀  Starting FastAPI backend on http://localhost:8000 ..."
cd "$PROJECT_ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Wait for backend to boot
sleep 3

echo ""
echo "🎨  Starting Streamlit frontend on http://localhost:8501 ..."
cd "$PROJECT_ROOT/frontend"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
echo "    Frontend PID: $FRONTEND_PID"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Backend  → http://localhost:8000"
echo "  API Docs → http://localhost:8000/docs"
echo "  Frontend → http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Press Ctrl+C to stop both services"
echo ""

# Trap to kill both on exit
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait
