#!/bin/bash
# Start both FastAPI and MCP server

echo "Starting Persona Agent System..."
echo "================================"

# Start FastAPI in background
echo "Starting FastAPI on port 8000..."
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Try to start MCP server (might fail if fastmcp not installed)
echo "Starting MCP server on port 8100..."
if python -c "import fastmcp" 2>/dev/null; then
    python -m src.mcp_server.server &
    MCP_PID=$!
    echo "✅ MCP Server running (PID: $MCP_PID)"
else
    echo "⚠️  MCP Server skipped (fastmcp not installed)"
    echo "   Install with: pip install fastmcp"
    MCP_PID=""
fi

echo "================================"
echo "✅ FastAPI running (PID: $API_PID)"
echo "================================"

# Wait for API process (primary service)
wait $API_PID

# If API exits, kill MCP if running
if [ -n "$MCP_PID" ]; then
    kill $MCP_PID 2>/dev/null
fi
exit $?
