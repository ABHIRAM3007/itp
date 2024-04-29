from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__, template_folder="templates")  # Creating a Flask app instance

todos = [{"task": "todo", "done": False}]  # Initializing a list to store todo items


@app.route("/")
def index():
    return render_template("index.html", todos=todos)  # Rendering the index.html template with todos


@app.route("/add", methods=["POST"])
def add():
    todo = request.form['todo']  # Getting the todo task from the form submission
    todos.append({"task": todo, "done": False})  # Adding the new todo item to the list
    return redirect(url_for("index"))  # Redirecting to the index page after adding the todo


@app.route("/edit/<int:index>", methods=["GET", "POST"])
def edit(index):
    todo = todos[index]  # Getting the todo item to be edited
    if request.method == "POST":
        todo['task'] = request.form["todo"]  # Updating the task with the new value from the form
        return redirect(url_for("index"))  # Redirecting to the index page after editing
    else:
        return render_template("edit.html", todo=todo, index=index)  # Rendering the edit.html template for editing


@app.route("/check/<int:index>")
def check(index):
    todos[index]['done'] = not todos[index]['done']  # Toggling the 'done' status of the todo item
    return redirect(url_for("index"))  # Redirecting to the index page after updating

@app.route("/delete/<int:index>")
def delete(index):
    del todos[index]  # Deleting the todo item from the list
    return redirect(url_for("index"))  # Redirecting to the index page after deletion


if __name__ == "__main__":
    app.run(debug=True)  # Running the Flask app in debug mode

