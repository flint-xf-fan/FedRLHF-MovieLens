#!/bin/bash

# Number of clients to run
NUM_CLIENTS=$(python -c 'import configure; print(configure.N_CLIENTS)')

# Function to start a client with retry logic
start_client() {
    local client_id=$1
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        echo "Starting client $client_id (Attempt $((retry_count+1)))"
        python start_client.py $client_id
        if [ $? -eq 0 ]; then
            echo "Client $client_id started successfully"
            return 0
        fi
        retry_count=$((retry_count+1))
        echo "Client $client_id failed to start. Retrying in 5 seconds..."
        sleep 5
    done

    echo "Failed to start client $client_id after $max_retries attempts"
    return 1
}

# Ensure the server is ready (you might need to implement this check)
echo "Waiting for server to be ready..."
sleep 10  # Adjust this based on your server startup time

# Loop to start multiple clients
for (( i=0; i<$NUM_CLIENTS; i++ ))
do
   start_client $i &
   sleep 5  # Increased delay between starting each client
done

# Wait for all background processes to finish
wait

echo "All clients have finished"