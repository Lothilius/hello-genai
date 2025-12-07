#!/bin/sh
set -e

# Start the Ollama server in the background
ollama serve &

# Store the PID of the server process
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for Ollama server to start..."
until curl -s -f http://localhost:11434/api/health >/dev/null 2>&1; do
    sleep 1
done


# Pull and run the llama3 model
echo "Pulling and running llama3 model..."
ollama pull llama3
ollama run llama3 &
MODEL_PID=$!

# Set trap to handle signals and forward them to child processes
trap 'kill -TERM $SERVER_PID $MODEL_PID 2>/dev/null' TERM INT

# Wait for server process to complete
wait $SERVER_PID

