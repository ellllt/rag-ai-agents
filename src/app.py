import os
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from state import State
from graph import compiled

app = Flask(__name__)

cors_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(',')
CORS(app, origins=cors_allowed_origins)

@app.route('/api/stream', methods=['POST'])
def stream():
    data = request.get_json()
    query = data.get('query', '')

    initial_state = State(query=query)
    result = compiled.invoke(initial_state)
    message = result["messages"][-1]

    def generate():
        yield f"data: {message}\n\n"

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


