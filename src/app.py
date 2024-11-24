from flask import Flask
from state import State
from graph import compiled

app = Flask(__name__)
@app.route('/')

def main():
    initial_state = State(query="")
    result = compiled.invoke(initial_state)
    return str(result["messages"][-1])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


