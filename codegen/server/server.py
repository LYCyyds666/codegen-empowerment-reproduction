import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from codegen.server.base import ServerConfig
import hydra
from codegen.environment import State
from codegen.generators import initialize_generator
from codegen.policies import Assistant

assistant: Assistant | None = None
assistant_name: str | None = None


class SimpleRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self):
        # Get content length to know how many bytes to read
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            # Parse the JSON data from the request body
            data = json.loads(post_data)

            # Check that the JSON is a list of strings
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                actions, _ = assistant(
                    [State(code=d, suggested_completion="", problem=None) for d in data]
                )

                response = [a.code for a in actions]

                self._set_headers()
            else:
                response = {"error": "Expected a JSON list of strings."}
                self._set_headers(status=400)
        except json.JSONDecodeError:
            response = {"error": "Invalid JSON provided."}
            self._set_headers(status=400)

        # Write the JSON response back to the client
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({"assistant_name": assistant_name}).encode("utf-8"))


@hydra.main(
    config_path="../configs/assistant_servers", config_name="eta4", version_base=None
)
def main(config: ServerConfig):
    global assistant
    global assistant_name

    assistant_name = config.assistant_name
    assistant_generator = initialize_generator(config.assistant.generator)
    assistant = Assistant(assistant_generator, config.assistant)

    server_address = ("", config.head_port)
    httpd = HTTPServer(server_address, SimpleRequestHandler)
    print(f"Starting server on port {config.head_port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
