import tkinter as tk
from tkinter import ttk
import asyncio
import threading
import base64
from ._belch_proxy import InterceptingProxy, send_raw_request

class BelchStudioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BelchStudio")
        self.root.geometry("1200x800")

        self.proxy = InterceptingProxy()
        self.proxy_thread = None
        self.flows = {}

        self.create_widgets()
        self.start_proxy()
        self.poll_flow_queue()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=5, pady=5)

        # -- Proxy Tab --
        self.proxy_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.proxy_frame, text="Proxy")
        
        self.flow_table = ttk.Treeview(
            self.proxy_frame,
            columns=("ID", "Method", "Host", "URL"),
            show="headings"
        )
        self.flow_table.heading("ID", text="ID")
        self.flow_table.heading("Method", text="Method")
        self.flow_table.heading("Host", text="Host")
        self.flow_table.heading("URL", text="URL")
        self.flow_table.pack(expand=True, fill="both")
        self.flow_table.bind("<<TreeviewSelect>>", self.on_flow_select)

        self.details_pane = ttk.PanedWindow(self.proxy_frame, orient=tk.VERTICAL)
        self.details_pane.pack(expand=True, fill="both")

        self.request_frame = ttk.Frame(self.details_pane)
        self.request_text = tk.Text(self.request_frame, height=10, width=80)
        self.request_text.pack(expand=True, fill="both")
        self.details_pane.add(self.request_frame, weight=1)

        self.response_frame = ttk.Frame(self.details_pane)
        self.response_text = tk.Text(self.response_frame, height=10, width=80)
        self.response_text.pack(expand=True, fill="both")
        self.details_pane.add(self.response_frame, weight=1)

        # -- Repeater Tab --
        self.repeater_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.repeater_frame, text="Repeater")

        self.repeater_request_text = tk.Text(self.repeater_frame, height=15)
        self.repeater_request_text.pack(fill="both", expand=True, padx=5, pady=2)
        
        self.send_button = ttk.Button(self.repeater_frame, text="Send", command=self.send_repeater_request)
        self.send_button.pack(pady=2)

        self.repeater_response_text = tk.Text(self.repeater_frame, height=15)
        self.repeater_response_text.pack(fill="both", expand=True, padx=5, pady=2)

        # -- Decoder Tab --
        self.decoder_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decoder_frame, text="Decoder")

        self.decoder_input_text = tk.Text(self.decoder_frame, height=10)
        self.decoder_input_text.pack(fill="both", expand=True, padx=5, pady=2)

        self.decode_button = ttk.Button(self.decoder_frame, text="Decode as Base64", command=self.decode_text)
        self.decode_button.pack(pady=2)
        self.encode_button = ttk.Button(self.decoder_frame, text="Encode as Base64", command=self.encode_text)
        self.encode_button.pack(pady=2)

        self.decoder_output_text = tk.Text(self.decoder_frame, height=10)
        self.decoder_output_text.pack(fill="both", expand=True, padx=5, pady=2)

    def start_proxy(self):
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.proxy.run())

        self.proxy_thread = threading.Thread(target=run_loop, daemon=True)
        self.proxy_thread.start()

    def poll_flow_queue(self):
        try:
            while not self.proxy.flow_queue.empty():
                flow = self.proxy.flow_queue.get_nowait()
                self.add_flow_to_table(flow)
        except Exception as e:
            print(f"Error polling queue: {e}")
        
        self.root.after(100, self.poll_flow_queue)

    def add_flow_to_table(self, flow):
        self.flows[flow.id] = flow
        self.flow_table.insert(
            "", "end",
            values=(flow.id, flow.request.method, flow.request.host, flow.request.path)
        )

    def on_flow_select(self, event):
        selected_item = self.flow_table.selection()
        if not selected_item:
            return

        item = self.flow_table.item(selected_item)
        flow_id = item["values"][0]
        flow = self.flows.get(flow_id)

        if flow:
            self.request_text.delete("1.0", tk.END)
            self.request_text.insert(tk.END, flow.request.text)
            
            self.response_text.delete("1.0", tk.END)
            if flow.response:
                self.response_text.insert(tk.END, flow.response.text)
            
            self.repeater_request_text.delete("1.0", tk.END)
            self.repeater_request_text.insert(tk.END, flow.request.text)

    def send_repeater_request(self):
        raw_request = self.repeater_request_text.get("1.0", tk.END)
        
        def _send():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(send_raw_request(raw_request))
            self.root.after(0, self.update_repeater_response, response)

        threading.Thread(target=_send, daemon=True).start()

    def update_repeater_response(self, response):
        self.repeater_response_text.delete("1.0", tk.END)
        self.repeater_response_text.insert(tk.END, response)

    def decode_text(self):
        try:
            input_data = self.decoder_input_text.get("1.0", tk.END).strip()
            decoded_data = base64.b64decode(input_data).decode('utf-8')
            self.decoder_output_text.delete("1.0", tk.END)
            self.decoder_output_text.insert(tk.END, decoded_data)
        except Exception as e:
            self.decoder_output_text.delete("1.0", tk.END)
            self.decoder_output_text.insert(tk.END, f"Error: {e}")

    def encode_text(self):
        try:
            input_data = self.decoder_input_text.get("1.0", tk.END).strip()
            encoded_data = base64.b64encode(input_data.encode('utf-8')).decode('utf-8')
            self.decoder_output_text.delete("1.0", tk.END)
            self.decoder_output_text.insert(tk.END, encoded_data)
        except Exception as e:
            self.decoder_output_text.delete("1.0", tk.END)
            self.decoder_output_text.insert(tk.END, f"Error: {e}")

    def on_closing(self):
        if self.proxy:
            self.proxy.shutdown()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = BelchStudioGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
