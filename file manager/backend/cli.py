import requests
from rich.console import Console
from rich.panel import Panel
import readline  # 新增：让输入支持光标移动、历史记录

API_URL = "http://127.0.0.1:8000/api/ask"
SESSION_ID = "cli-session"

console = Console()

def chat():
    console.print("[bold green]进入终端聊天模式（输入 exit 退出）[/bold green]\n")
    while True:
        try:
            query = input("你: ")  # 现在支持左右方向键、退格、上下历史
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold red]结束对话[/bold red]")
            break

        if query.strip().lower() in ["exit", "quit"]:
            console.print("[bold red]结束对话[/bold red]")
            break

        try:
            resp = requests.post(API_URL, data={"query": query, "session_id": SESSION_ID})
            if resp.status_code == 200:
                data = resp.json()
                ans = data.get("answer", "（无返回）")
                console.print(Panel.fit(ans, title="AI", border_style="cyan"))
            else:
                console.print(f"[red]错误 {resp.status_code}[/red]: {resp.text}")
        except Exception as e:
            console.print(f"[red]请求失败[/red]: {e}")

if __name__ == "__main__":
    chat()

