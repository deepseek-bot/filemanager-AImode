# filemanager-AImode
An AI version of your personal database. Upload files, and the built‑in model will answer your questions based only on that data. Simple demo with Python and Docker, easy to run and experiment.

📢Note:Instructions are based on Ubuntu 24.04. Commands may differ on other Linux systems. 

⚠️ Notification
This project relies on GPU acceleration.  
- If you plan to run it on a VPS, it is recommended to use a VDS with GPU support or attach a dedicated GPU.  
- Running in CPU‑only mode may cause memory to accumulate without proper release, eventually exhausting system resources and crashing the machine.  
- GPUs, by contrast, provide built‑in memory management for model execution, making them more stable and efficient for long‑running tasks.  

🔧 Features
- Personal AI Database: Upload files, and the built‑in model answers questions only based on your data.  
- File Monitoring: Automatically watches a folder for new uploads.  
- Embeddings with Ollama: Converts text into vector embeddings.  
- Chroma Storage: Stores embeddings for fast semantic search.  
- Lightweight Demo: Simple design, easy to run with Python or Docker.  

👀Usage:
git clone https://github.com/deepseek-bot/filemanager-AImode.git 

1. Install Docker  
   `
   sudo apt install docker -y
   `
2. Install Docker Compose plugin  
   `
   sudo apt install docker-compose-plugin -y
   `
3. Create network  
   `
   docker network create ollama-net
   `
4. Start Ollama container  
   `
   docker compose up
   `
5. Pull Ollama model  
   `
   docker exec -it ollama bash -c "ollama pull qwen2.5:3b"
   `
6. Start filemanager container  
   `
   cd filemanager
   docker compose up
   `
ps:container name default: ollama and personal_rag_backend
model default:
- embedding model: nomic-embed-text
- language model defalt: qwen2.5:3b

⚒️Use steps(demo):
Run file monitoring:  
`
docker exec -it personalragbackend bash -c "python3 watch_uploads.py"
`

- When you run watch_uploads.py, the script occupies the current terminal session to monitor files.  
- Because the session is blocked, you cannot upload files via command line in the same window.  
- Instead, use your SSH client’s built‑in file manager (e.g. drag‑and‑drop or file panel) to place files into the uploads/data folder.  
- The script will automatically detect and process new files once they appear.  
- After uploading files, press Ctrl+C to quit.  

Interact with AI inside the container:  
`
docker exec -it personalragbackend bash -c "python3 cli.py"
`

Run in background (optional):  
`
docker exec -it personalragbackend bash -c "nohup python3 watch_uploads.py &"
`
The process will run in the background.  
To stop it:  
`
ps aux | grep watch_uploads.py
`
`

kill <PID>
`

⚡Personalized usage:
If you want to change the Ollama model, simply update the model name in both `docker-compose.yml` and `main.py`, and make sure they remain consistent.

